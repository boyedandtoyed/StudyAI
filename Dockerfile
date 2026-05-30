FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main_fastapi.py gg.py ./

RUN mkdir -p demo_pdfs

EXPOSE 8000

CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
