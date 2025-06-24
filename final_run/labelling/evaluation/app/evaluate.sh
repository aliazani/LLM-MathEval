#!/usr/bin/env bash
set -euo pipefail

# Configuration
readonly MATHCHAT_REPO_URL="https://github.com/Zhenwen-NLP/MathChat.git"
readonly MATHCHAT_REPO="/app/MathChat"
readonly FOLLOW_UP_JSON="${MATHCHAT_REPO}/MathChat Benchmark/follow_up.jsonl"

readonly QUESTIONS_CSV="/app/questions.csv"
readonly QA_JSONL="/app/qa.jsonl"
readonly SPANS_CSV="/app/vllm_spans.csv"

readonly MAX_QUESTIONS=1500

# Environment variables with defaults
PHOENIX_URL=${PHOENIX_URL:-"http://phoenix:6006"}
PHOENIX_PROJECT=${PHOENIX_PROJECT:-"vllm-custom-server-final"}
OLLAMA_HOST=${OLLAMA_HOST:-"ollama"}
OLLAMA_PORT=${OLLAMA_PORT:-"11434"}

log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
}

log_step() {
    log "🔄  $*"
}

log_success() {
    log "✅  $*"
}

log_error() {
    log "❌  $*" >&2
}

# Wait for services to be ready
wait_for_services() {
    log_step "Waiting for services to be ready..."
    
    # Wait for Phoenix
    log "Waiting for Phoenix at ${PHOENIX_URL}..."
    until curl -sf "${PHOENIX_URL}" > /dev/null 2>&1; do
        echo "Waiting for Phoenix..."
        sleep 2
    done
    log_success "Phoenix is ready"
    
    # Wait for Ollama
    log "Waiting for Ollama at http://${OLLAMA_HOST}:${OLLAMA_PORT}..."
    until curl -sf "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; do
        echo "Waiting for Ollama..."
        sleep 2
    done
    log_success "Ollama is ready"
}

# Clone MathChat repository
clone_mathchat() {
    log_step "Setting up MathChat repository..."
    
    if [[ ! -d "$MATHCHAT_REPO" ]]; then
        log "Cloning MathChat repository..."
        git clone --depth 1 "$MATHCHAT_REPO_URL" "$MATHCHAT_REPO"
        log_success "Repository cloned"
    else
        log "Repository already exists"
    fi
    
    if [[ ! -f "$FOLLOW_UP_JSON" ]]; then
        log_error "Follow-up data file not found: $FOLLOW_UP_JSON"
        return 1
    fi
}

# Extract Q&A pairs from MathChat data
extract_qa_pairs() {
    log_step "Extracting Q&A pairs from MathChat data..."
    
    python3 /app/extract_questions_and_answers.py \
        --input "$FOLLOW_UP_JSON" \
        --questions_output "$QUESTIONS_CSV" \
        --qa_output "$QA_JSONL" \
        --max "$MAX_QUESTIONS"
    
    if [[ ! -f "$QA_JSONL" ]]; then
        log_error "Failed to create Q&A file: $QA_JSONL"
        return 1
    fi
    
    local qa_count
    qa_count=$(wc -l < "$QA_JSONL")
    log_success "Extracted ${qa_count} Q&A pairs"
}

# Download CSV data from Phoenix
download_csv_data() {
    log_step "Downloading CSV data from Phoenix..."
    
    PHOENIX_URL="$PHOENIX_URL" \
    PHOENIX_PROJECT="$PHOENIX_PROJECT" \
    CSV_OUTPUT="$SPANS_CSV" \
    python3 /app/download_data.py
    
    if [[ ! -f "$SPANS_CSV" ]]; then
        log_error "Failed to download CSV data"
        return 1
    fi
    
    local csv_rows
    csv_rows=$(wc -l < "$SPANS_CSV")
    log_success "Downloaded CSV with ${csv_rows} rows"
}

# Run the evaluation
run_evaluation() {
    log_step "Running answer evaluation..."
    
    # Ensure output directory exists
    mkdir -p /app/output
    
    python3 /app/answer_evaluator.py \
        --csv "$SPANS_CSV" \
        --jsonl "$QA_JSONL" \
        --host "$OLLAMA_HOST" \
        --port "$OLLAMA_PORT" \
        --output-csv "/app/output/results.csv" \
        --summary-file "/app/output/summary.json"
    
    if [[ -f "/app/output/summary.json" ]]; then
        log_success "Evaluation complete!"
        log "Results summary:"
        cat /app/output/summary.json
    else
        log_error "Evaluation failed - no summary file generated"
        return 1
    fi
}

# Main execution
main() {
    log_step "Starting LLM answer evaluation pipeline..."
    
    wait_for_services
    clone_mathchat
    extract_qa_pairs
    download_csv_data
    run_evaluation
    
    log_success "Pipeline completed successfully!"
    log "Output files:"
    log "  - Detailed results: /app/output/results.csv"
    log "  - Summary: /app/output/summary.json"
    log "  - Q&A pairs: /app/qa.jsonl"
}

main "$@"
