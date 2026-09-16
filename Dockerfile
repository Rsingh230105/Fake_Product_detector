FROM python:3.10.13-slim

# Python ko .pyc files na banane do, aur logs turant screen pe dikhein (buffer na ho)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System-level dependencies (pip se nahi aate, apt se install hote hain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libzbar0 \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pehle sirf requirement files copy karo (layer caching ke liye — agla topic mein detail)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install python-magic==0.4.27
    
# Ab poora project copy karo (.dockerignore jo exclude karega, wo skip hoga)
COPY . .
RUN chmod +x entrypoint.sh

# Security: root user se mat chalao — non-root user banao
RUN useradd -m appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]