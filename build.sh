#!/usr/bin/env bash
set -e

echo "==> Python version check"
python --version

echo "==> Installing system packages (tesseract, zbar)"
apt-get install -y tesseract-ocr libzbar0 2>/dev/null || echo "apt-get skipped (not root or already installed)"

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Collecting static files"
cd webapp
python manage.py collectstatic --noinput

echo "==> Running database migrations"
python manage.py migrate --noinput

echo "==> Build complete"
