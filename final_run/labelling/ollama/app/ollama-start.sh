#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting Ollama server in background…"
echo "📋 Current OLLAMA_HOST: ${OLLAMA_HOST:-not set}"

# Start Ollama server
ollama serve &
SERVER_PID=$!

# Give the server a moment to start
sleep 2

# Try different health check approaches
echo "⏳ Waiting for Ollama API…"

max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    
    # Try multiple endpoints
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama server is ready!"
        break
    elif curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama server is ready!"
        break
    elif curl -sf http://0.0.0.0:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama server is ready!"
        break
    fi
    
    echo "…still waiting for Ollama… (attempt $attempt/$max_attempts)"
    
    # Show what's happening
    if [ $attempt -eq 5 ]; then
        echo "🔍 Checking network status…"
        netstat -tlnp 2>/dev/null | grep 11434 || true
        ps aux | grep ollama || true
    fi
    
    sleep 2
done

if [ $attempt -ge $max_attempts ]; then
    echo "❌ Ollama server failed to start after $max_attempts attempts"
    echo "🔍 Final diagnostics:"
    netstat -tlnp 2>/dev/null | grep 11434 || true
    ps aux | grep ollama || true
    curl -v http://localhost:11434/api/tags 2>&1 || true
    exit 1
fi

echo "📥 Pulling llama3.3 model…"
ollama pull llama3.3

echo "✅ Model downloaded. Ollama ready!"

# Keep the server running
wait "$SERVER_PID"
