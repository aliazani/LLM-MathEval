# Ollama container - just runs Ollama with llama3.3
FROM ollama/ollama:latest

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Persist models & metadata
VOLUME ["/root/.ollama"]

# Configure Ollama server bindings
ENV OLLAMA_HOST=0.0.0.0

# Copy startup script
COPY ./app/ollama-start.sh /usr/local/bin/ollama-start.sh
RUN chmod +x /usr/local/bin/ollama-start.sh

EXPOSE 11434

ENTRYPOINT ["/usr/local/bin/ollama-start.sh"]
