#!/usr/bin/env bash
set -euo pipefail

# Entrypoint for fine-tuning Mistral 7B with LoRA mathchat adapter
# Adjust environment variables as needed or override at runtime.

# Required: Hugging Face token
: "${HF_TOKEN:?Environment variable HF_TOKEN must be set}"  # exit if not set

# Default values (can override by exporting these variables)
DATASET_URL="${DATASET_URL:-1nkAXAL9EpmDiceoV_qv6M00Lj0LA7MKX}"
DATASET_PATH="${DATASET_PATH:-data/mathchat.json}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.3}"
TOKENIZER_NAME="${TOKENIZER_NAME:-mistralai/Mistral-7B-Instruct-v0.3}"
OUTPUT_DIR="${OUTPUT_DIR:-aliazn/mathchat-mistral}"
SPLIT_RATIO="${SPLIT_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-3e-5}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
SEED="${SEED:-42}"
TRUST_REMOTE_CODE_FLAG=""
if [ "${TRUST_REMOTE_CODE:-false}" = "true" ]; then
  TRUST_REMOTE_CODE_FLAG="--trust_remote_code"
fi

# Echo configuration
echo "Running fine-tune with settings:"
echo "  HF_TOKEN        = [REDACTED]"
echo "  DATASET_URL     = $DATASET_URL"
echo "  DATASET_PATH    = $DATASET_PATH"
echo "  MODEL_NAME      = $MODEL_NAME"
echo "  TOKENIZER_NAME  = $TOKENIZER_NAME"
echo "  OUTPUT_DIR      = $OUTPUT_DIR"
echo "  SPLIT_RATIO     = $SPLIT_RATIO"
echo "  MAX_LENGTH      = $MAX_LENGTH"
echo "  EPOCHS          = $EPOCHS"
echo "  BATCH_SIZE      = $BATCH_SIZE"
echo "  GRAD_ACCUM      = $GRAD_ACCUM"
echo "  LR              = $LR"
echo "  LORA_R          = $LORA_R"
echo "  LORA_ALPHA      = $LORA_ALPHA"
echo "  LORA_DROPOUT    = $LORA_DROPOUT"
echo "  SEED            = $SEED"
echo "  TRUST_REMOTE_CODE = ${TRUST_REMOTE_CODE:-false}"

echo "Starting training..."

# Execute fine-tuning script
python finetune.py \
  --dataset_url "$DATASET_URL" \
  --dataset_path "$DATASET_PATH" \
  --model_name "$MODEL_NAME" \
  --tokenizer_name "$TOKENIZER_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --split_ratio "$SPLIT_RATIO" \
  --max_length "$MAX_LENGTH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --seed "$SEED" \
  $TRUST_REMOTE_CODE_FLAG \
  --hf_token "$HF_TOKEN"

echo "Fine tuning finished."
