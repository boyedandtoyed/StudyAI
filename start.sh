#!/bin/bash
cd /home/blinded-christi/Desktop/fast_api_study_ai/StudyAI
exec venv/bin/uvicorn main_fastapi:app --host 0.0.0.0 --port 8002
