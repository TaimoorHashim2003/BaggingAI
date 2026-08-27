import json
import subprocess
from pathlib import Path
from ultralytics import YOLO
import whisper
import math
import os

BASE_DIR = Path(__file__).parent.parent
JOBS_DIR = BASE_DIR / "uploads" / "jobs"


def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def update_meta(job_dir: Path, **kwargs):
    meta_path = job_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(kwargs)
    meta_path.write_text(json.dumps(meta))


def get_video_info(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", str(path)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(p.stdout)
    width = int(info["streams"][0]["width"])
    height = int(info["streams"][0]["height"])
    duration = float(info["format"]["duration"])
    return width, height, duration


def extract_audio(input_path, audio_path):
    run(["ffmpeg", "-y", "-i", str(input_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)])


def transcribe_whisper(audio_path, model_name="small"):
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), word_timestamps=False)
    return result.get("segments", [])


def sample_frames(input_path, frames_dir, fps=2.0):
    frames_dir.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(input_path), "-vf", f"fps={fps}", str(frames_dir / "frame_%06d.jpg")])
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    return frames, fps


def detect_objects_on_frames(model, frames, labels_of_interest):
    detections = []
    for idx, frame in enumerate(frames):
        results = model.predict(str(frame), imgsz=640, conf=0.25, verbose=False)
        r = results[0]
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = r.names[cls]
            if label in labels_of_interest:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append((idx, label, xyxy, conf))
    return detections


def frame_idx_to_time(idx, fps):
    return idx / fps


def build_intervals_from_detections(detections, fps, padding, max_len):
    times = [frame_idx_to_time(d[0], fps) for d in detections]
    intervals = []
    for t in times:
        start = max(0, t - padding)
        end = t + padding
        dur = end - start
        if dur > max_len:
            mid = (start + end) / 2
            start = max(0, mid - max_len/2)
            end = start + max_len
        intervals.append((start, end))
    intervals.sort()
    merged = []
    for s,e in intervals:
        if not merged or s > merged[-1][1] + 0.5:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(float(s), float(e)) for s,e in merged]


def write_srt(segments, path):
    # segments from whisper have start, end, text
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        def fmt(t):
            h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t - int(t)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lines.append(f"{i}\n{fmt(start)} --> {fmt(end)}\n{text}\n")
    path.write_text("\n".join(lines))


def clip_srt_for_interval(segments, start, end):
    # return srt text only for segments that intersect [start,end]
    lines = []
    idx = 1
    def fmt(t):
        h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    for seg in segments:
        a = seg["start"]
        b = seg["end"]
        if b < start or a > end:
            continue
        text = seg["text"].strip()
        # clamp times to interval start for display consistency
        s = max(a, start) - start
        e = min(b, end) - start
        # convert to absolute times relative to 0 for the clip's SRT: we want typical SRT absolute time; ffmpeg expects absolute timeline matching clip's timestamps when burning with -ss -to? Simpler: shift to small window starting at 0
        # For burning with ffmpeg using subtitles filter, it's easier to create SRT with absolute times matching original video times. We'll instead create an SRT with absolute timestamps (original time)
        lines.append(f"{idx}\n{fmt(a)} --> {fmt(b)}\n{text}\n")
        idx += 1
    return "\n".join(lines)


def cut_and_crop_clip(input_path, out_path, start, end, crop_box, out_w=1080, out_h=1920, srt_path=None):
    filters = []
    if crop_box is not None:
        x_ctr, y_ctr, cw, ch = crop_box
        x = max(0, int(x_ctr - cw/2))
        y = max(0, int(y_ctr - ch/2))
        crop_filter = f"crop={int(cw)}:{int(ch)}:{x}:{y}"
        filters.append(crop_filter)
    filters.append(f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease")
    filters.append(f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2")
    vf = ",".join(filters)

    # If srt_path is provided, burn subtitles using subtitles filter which needs file path
    if srt_path:
        vf = vf + f",subtitles={srt_path}"

    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(input_path), "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-c:a", "aac", "-b:a", "128k", str(out_path)]
    run(cmd)


def process_job(job_id, input_path, meta):
    job_dir = JOBS_DIR / job_id
    try:
        update_meta(job_dir, status="processing", progress=0)
    except Exception:
        pass

    input_path = Path(input_path)
    out_clips_dir = job_dir / "clips"
    out_clips_dir.mkdir(parents=True, exist_ok=True)

    # video info
    vw, vh, duration = get_video_info(input_path)
    update_meta(job_dir, progress=5)

    # audio
    audio_path = job_dir / "audio.wav"
    extract_audio(input_path, audio_path)
    update_meta(job_dir, progress=15)

    # transcribe
    segments = transcribe_whisper(audio_path, model_name="small")
    update_meta(job_dir, progress=35)

    # write full srt
    srt_path = job_dir / "transcript.srt"
    write_srt(segments, srt_path)

    # sample frames
    frames_dir = job_dir / "frames"
    frames, fps = sample_frames(input_path, frames_dir, fps=2.0)
    update_meta(job_dir, progress=50)

    # object detection
    labels = [l.strip() for l in meta.get("labels","").split(",") if l.strip()]
    detections = []
    if labels:
        model = YOLO(meta.get("yolo_model", "yolov8n.pt"))
        detections = detect_objects_on_frames(model, frames, labels)
    update_meta(job_dir, progress=65)

    intervals = []
    if detections:
        intervals = build_intervals_from_detections(detections, fps, meta.get("padding",4.0), meta.get("max_len",30.0))

    # transcript keywords
    kws = [k.strip().lower() for k in str(meta.get("keywords","")).split(",") if k.strip()]
    if kws:
        for seg in segments:
            text = seg["text"].lower()
            if any(k in text for k in kws):
                s = max(0, seg["start"] - meta.get("padding",4.0))
                e = min(duration, seg["end"] + meta.get("padding",4.0))
                dur = e - s
                if dur > meta.get("max_len",30.0):
                    mid = (s+e)/2
                    s = max(0, mid - meta.get("max_len",30.0)/2)
                    e = s + meta.get("max_len",30.0)
                intervals.append((s,e))

    # merge intervals
    intervals.sort()
    merged = []
    for s,e in intervals:
        if not merged or s > merged[-1][1] + 0.5:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    merged = [(float(s), float(e)) for s,e in merged]

    update_meta(job_dir, progress=75)

    # create clips with per-clip srt
    created = []
    for i,(s,e) in enumerate(merged, start=1):
        out_file = out_clips_dir / f"clip_{i:03d}.mp4"
        # find detection boxes in interval
        dets_in = [d for d in detections if frame_idx_to_time(d[0], fps) >= s - 0.1 and frame_idx_to_time(d[0], fps) <= e + 0.1]
        crop_box = None
        if dets_in:
            best = max(dets_in, key=lambda x: (x[2][2]-x[2][0])*(x[2][3]-x[2][1]))
            x1,y1,x2,y2 = best[2]
            cw = x2 - x1; ch = y2 - y1
            pad_factor = 2.2
            cw_e = min(vw, cw * pad_factor)
            ch_e = min(vh, ch * pad_factor)
            w_aspect, h_aspect = map(int, str(meta.get("aspect","9:16")).split(":"))
            target_ar = w_aspect / h_aspect
            if (cw_e / ch_e) >= target_ar:
                crop_w = cw_e
                crop_h = crop_w / target_ar
            else:
                crop_h = ch_e
                crop_w = crop_h * target_ar
            x_ctr = (x1 + x2) / 2
            y_ctr = (y1 + y2) / 2
            crop_w = min(crop_w, vw)
            crop_h = min(crop_h, vh)
            crop_box = (x_ctr, y_ctr, crop_w, crop_h)
        else:
            w_aspect, h_aspect = map(int, str(meta.get("aspect","9:16")).split(":"))
            target_ar = w_aspect / h_aspect
            if (vw / vh) >= target_ar:
                crop_h = vh
                crop_w = crop_h * target_ar
            else:
                crop_w = vw
                crop_h = crop_w / target_ar
            crop_box = (vw/2, vh/2, crop_w, crop_h)

        # create per-clip srt
        clip_srt_text = clip_srt_for_interval(segments, s, e)
        clip_srt_path = job_dir / f"clip_{i:03d}.srt"
        clip_srt_path.write_text(clip_srt_text)

        cut_and_crop_clip(input_path, out_file, s, e, crop_box, out_w=1080, out_h=1920, srt_path=str(clip_srt_path))
        created.append(str(out_file.name))

    update_meta(job_dir, status="done", progress=100, clips=created)

    # cleanup frames to save space
    try:
        for f in (job_dir / "frames").glob("*.jpg"):
            f.unlink()
    except Exception:
        pass

    return True
