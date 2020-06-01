#!/bin/bash
exec gunicorn -b :$SERVICE_PORT --workers $SERVICE_WORKERS --threads $SERVICE_THREADS --timeout $GUNICORN_TIMEOUT --access-logfile - --error-logfile - --log-level=debug app:app
exit