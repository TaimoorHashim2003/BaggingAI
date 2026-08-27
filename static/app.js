const form = document.getElementById('upload-form');
const statusDiv = document.getElementById('status');
const jobIdDiv = document.getElementById('job-id');
const progressDiv = document.getElementById('progress');
const clipsDiv = document.getElementById('clips');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('file');
  if (!fileInput.files.length) return alert('Select a file');
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('labels', document.getElementById('labels').value);
  fd.append('keywords', document.getElementById('keywords').value);
  fd.append('max_len', document.getElementById('max_len').value);
  fd.append('padding', document.getElementById('padding').value);
  fd.append('aspect', document.getElementById('aspect').value);

  const res = await fetch('/upload', { method: 'POST', body: fd });
  const data = await res.json();
  const jobId = data.job_id;
  statusDiv.style.display = 'block';
  jobIdDiv.innerText = 'Job ID: ' + jobId;
  pollJob(jobId);
});

async function pollJob(jobId) {
  progressDiv.innerText = 'Starting...';
  const iv = setInterval(async () => {
    const res = await fetch('/jobs/' + jobId);
    if (res.status === 404) {
      progressDiv.innerText = 'Job not found';
      clearInterval(iv);
      return;
    }
    const meta = await res.json();
    progressDiv.innerText = `Status: ${meta.status} - ${meta.progress}%`;
    if (meta.clips && meta.clips.length) {
      clipsDiv.innerHTML = '<h3>Clips</h3>' + meta.clips.map(c => `<div><a href="/clips/${jobId}/clips/${c}" target="_blank">${c}</a></div>`).join('');
    }
    if (meta.status === 'done' || meta.status === 'failed') {
      clearInterval(iv);
    }
  }, 3000);
}
