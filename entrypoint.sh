#!/usr/bin/env bash
set -o errexit

echo "==> Downloading ML model from S3 (if not present)..."
if [ ! -f "models/mobilenet_v2_food_production.keras" ]; then
    python -c "
import boto3, os
s3 = boto3.client('s3', region_name=os.environ.get('AWS_S3_REGION_NAME', 'ap-south-1'))
bucket = os.environ['AWS_STORAGE_BUCKET_NAME']
s3.download_file(bucket, 'models/mobilenet_v2_food_production.keras', 'models/mobilenet_v2_food_production.keras')
print('Model downloaded successfully.')
"
fi

echo "==> Collecting static files..."
python webapp/manage.py collectstatic --no-input

echo "==> Running migrations..."
python webapp/manage.py migrate --no-input

echo "==> Starting Gunicorn..."
exec gunicorn --chdir webapp ai_product_verification_system.wsgi:application \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${WEB_CONCURRENCY:-3} \
    --timeout 120 \
    --log-level info