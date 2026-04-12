#!/usr/bin/env bash
# build.sh - Production build script for Render deployment

set -o errexit  # Exit on any error

echo "🚀 Starting build process..."

# Update pip to latest version
echo "📦 Updating pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create models directory for ML model storage
echo "📁 Creating models directory..."
mkdir -p models

# Collect static files for Django
echo "🎨 Collecting static files..."
python webapp/manage.py collectstatic --no-input

# Run database migrations
echo "🗄️ Running database migrations..."
python webapp/manage.py migrate --no-input

echo "✅ Build completed successfully!"
echo "🎯 Ready for deployment on Render"