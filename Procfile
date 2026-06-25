web: gunicorn --chdir webapp food_detection.wsgi:application --bind 0.0.0.0:$PORT --workers $WEB_CONCURRENCY --timeout 120 --log-level info
