FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1

# OCR + PDF rasterization backends (used by the flexible OCR provider).
# tesseract-ocr-vie/eng for Vietnamese+English; poppler-utils for scanned PDFs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
