#!/usr/bin/env bash
# Run development server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
