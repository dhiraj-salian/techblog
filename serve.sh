#!/bin/bash
# Lightweight static file server for Medium content
# Usage: ./serve.sh [port]
# Default port: 8080

PORT=${1:-8080}
cd /home/dhiraj/.openclaw/workspace/medium/posts

echo "Starting Medium server on port $PORT..."
python3 -m http.server $PORT --bind 127.0.0.1