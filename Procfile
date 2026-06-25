web: gunicorn --chdir webapp ai_product_verification_system.wsgi:application --bind 0.0.0.0:$PORT --workers $WEB_CONCURRENCY --timeout 120 --log-level info
