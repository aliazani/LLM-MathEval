FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# ---- OS packages ----
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    curl ca-certificates wget   \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# ---- Python virtualenv ----
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

RUN pip install --upgrade pip

# ---- Core runtime deps ----
RUN pip install \
    torch==2.6

RUN pip install flashinfer-python \
    -i https://flashinfer.ai/whl/cu126/torch2.6


# Install vLLM
RUN pip install \
    vllm \
    transformers \
    huggingface_hub \
    peft \
    sentencepiece \
    "protobuf<4.0.0"


# ---- Workdir & code ----
WORKDIR /app
RUN mkdir -p /app/emissions_logs
COPY ./app /app
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]