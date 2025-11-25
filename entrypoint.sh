#!/bin/bash
set -e

if [ "$1" = "train" ]; then
    echo "Running training pipeline..."
    python -m src.pipeline.pipeline
elif [ "$1" = "serve" ]; then
    echo "Starting API server..."
    uvicorn app.app:app --host 0.0.0.0 --port $PORT
else
    echo "Unknown command: $1"
    echo "Use: train | serve"
    exec "$@"
fi
