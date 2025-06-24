# Dockerfile.phoenix
FROM python:3.10-slim

# ── system deps so pyarrow / pandas can build ───────────────────────────────
RUN apt-get update && apt-get install -y build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Phoenix with Parquet support + helpers
RUN pip install --no-cache-dir \
      "arize-phoenix[parquet]" \
      nest-asyncio \
      pyarrow \
      pandas

# copy your code (including phoenix_startup.py)
COPY ./app /app

# make sure the data sub-dir exists (works for first run and for CI builds)
RUN mkdir -p /app/.phoenix/exports

EXPOSE 6006
CMD ["python", "/app/phoenix_startup.py"]

