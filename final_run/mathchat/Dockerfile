FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget ca-certificates unzip jq \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    sysstat \
    && rm -rf /var/lib/apt/lists/*

# Set default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create and activate virtual environment
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support and FlashInfer for performance
RUN pip install torch==2.6
RUN pip install flashinfer-python \
    -i https://flashinfer.ai/whl/cu126/torch2.6

# Install vLLM and its dependencies
RUN pip install \
    transformers \
    huggingface_hub \
    peft \
    vllm \
    sentencepiece \
    "protobuf<4.0.0"


# Install energy monitoring and proxy dependencies
RUN pip install \
    codecarbon \
    fastapi \
    uvicorn[standard] \
    gunicorn \
    httpx \
    pydantic \
    requests

# Install OpenTelemetry, Phoenix, and semantic conventions dependencies
RUN pip install \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-instrumentation-fastapi \
    openinference-instrumentation \
    openinference-semantic-conventions \
    arize-phoenix \
    arize-phoenix-otel

# Create working directory
WORKDIR /app

# Create emissions log directory to prevent CodeCarbon errors
RUN mkdir -p /app/emissions_logs

# Copy app code
COPY ./app /app

# Copy and set entrypoint
# Assuming you have an entrypoint.sh that runs `uvicorn app.server:app ...`
COPY ./app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV VLLM_SKIP_LORA_TOKENIZER_CHECK=1

EXPOSE 8000

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
