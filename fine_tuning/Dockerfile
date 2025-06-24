FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Set environment to non-interactive
# ENV DEBIAN_FRONTEND=noninteractive

# Basic system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create and activate virtual environment
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip and install packages inside venv
RUN pip install --upgrade pip
RUN pip install torch==2.6.0 deepspeed datasets peft accelerate bitsandbytes gdown sentencepiece protobuf==3.20 huggingface_hub
RUN pip install "transformers[torch]"


# Set the working directory
WORKDIR /app

# Copy project files
COPY ./app /app

# Default command
CMD ["/app/entrypoint.sh"]
