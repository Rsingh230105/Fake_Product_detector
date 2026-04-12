#!/usr/bin/env bash
# build.sh — Render build script
set -o errexit

echo "==> Installing system dependencies..."
apt-get install -y tesseract-ocr libzbar0 2>/dev/null || true

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Creating models directory..."
mkdir -p models

echo "==> Collecting static files..."
python webapp/manage.py collectstatic --no-input

echo "==> Running migrations..."
python webapp/manage.py migrate --no-input

echo "==> Build complete."
