# Evaluation container - minimal base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
RUN pip install arize-phoenix pandas requests jsonlines

# Copy evaluation scripts
COPY ./app/download_data.py /app/download_data.py
COPY ./app/extract_questions_and_answers.py /app/extract_questions_and_answers.py
COPY ./app/answer_evaluator.py /app/answer_evaluator.py
COPY ./app/evaluate.sh /app/evaluate.sh

# Make scripts executable
RUN chmod +x /app/evaluate.sh

# Default command
CMD ["/app/evaluate.sh"]
