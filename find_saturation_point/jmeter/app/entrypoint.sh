#!/usr/bin/env bash

# =============================================================================
# SCRIPT CONFIGURATION
#
# This script automates the process of setting up and running a JMeter
# performance test and generating its HTML report.
#
# Exit on error, undefined variables, or pipe failures
set -euo pipefail
# =============================================================================

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
readonly APP_DIR="/app"
readonly MATHCHAT_REPO_URL="https://github.com/Zhenwen-NLP/MathChat.git"
readonly MATHCHAT_REPO="${APP_DIR}/MathChat"

readonly QUESTIONS_CSV="${APP_DIR}/questions.csv"
readonly FOLLOW_UP_JSON="${MATHCHAT_REPO}/MathChat Benchmark/follow_up.jsonl"
readonly EXTRACT_SCRIPT="${APP_DIR}/extract_questions_and_answers.py"
readonly JMETER_PLAN="${APP_DIR}/generate-plan.jmx"

readonly RESULTS_FILE="${APP_DIR}/results_run.jtl"
readonly LOG_FILE="${APP_DIR}/jmeter_run.log"
readonly REPORT_DIR="${APP_DIR}/report"

readonly MAX_QUESTIONS=1500

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Logs a message with a timestamp.
log() {
    # ISO 8601 format timestamp for better sorting and universal parsing.
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
}

# Logs an informational message, indented for readability.
log_info() {
    log "    $*"
}

# Logs a success message with a checkmark.
log_success() {
    log "✅  $*"
}

# Logs an error message with a cross mark to standard error.
log_error() {
    log "❌  $*" >&2
}

# Logs a step in the process with a refresh icon.
log_step() {
    log "🔄  $*"
}

# Removes a list of files if they exist.
cleanup_files() {
    local files_to_remove=("$@")
    for file in "${files_to_remove[@]}"; do
        if [[ -f "$file" ]]; then
            rm -f "$file"
            log_info "Removed: $file"
        fi
    done
}

# =============================================================================
# CORE SCRIPT FUNCTIONS
# =============================================================================

cleanup_previous_outputs() {
    log_step "Cleaning up outputs from previous runs..."

    local files_to_clean=(
        "${APP_DIR}/jmeter.log"
        "${RESULTS_FILE}"
        "${LOG_FILE}"
        "${APP_DIR}/results_"*.jtl
        "${APP_DIR}/results_"*.csv
        "${APP_DIR}/jmeter_"*.log
    )

    cleanup_files "${files_to_clean[@]}"
    
    log_success "Cleanup complete."
}

verify_prerequisites() {
    log_step "Verifying prerequisites..."

    # 1. Check for required command-line tools.
    local required_commands=(git python3 jmeter curl unzip nc)
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: '$cmd'. Please install it to proceed."
            return 1
        fi
    done

    # 2. Clone MathChat repo if missing.
    if [[ ! -d "$MATHCHAT_REPO" ]]; then
        log_info "MathChat repo not found at '${MATHCHAT_REPO}'. Cloning..."
        if ! git clone --depth 1 "$MATHCHAT_REPO_URL" "$MATHCHAT_REPO"; then
            log_error "Failed to clone MathChat repo."
            return 1
        fi
        log_success "Successfully cloned repository."
    else
        log_info "MathChat repository already exists."
    fi

    # 3. Ensure essential files exist.
    local required_files=(
        "$FOLLOW_UP_JSON"
        "$EXTRACT_SCRIPT"
        "$JMETER_PLAN"
    )
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            return 1
        fi
    done

    log_success "All prerequisites are satisfied."
}

extract_questions() {
    log_step "Extracting questions from MathChat Benchmark..."

    # Regenerate only if source is newer.
    if [[ -f "$QUESTIONS_CSV" && "$FOLLOW_UP_JSON" -ot "$QUESTIONS_CSV" ]]; then
        log_info "Using existing, up-to-date questions file: ${QUESTIONS_CSV}"
        return 0
    elif [[ -f "$QUESTIONS_CSV" ]]; then
        log_info "Source file is newer. Regenerating questions..."
        rm -f "$QUESTIONS_CSV"
    fi

    if ! python3 "$EXTRACT_SCRIPT" \
        --input "$FOLLOW_UP_JSON" \
        --questions_output "$QUESTIONS_CSV" \
        --qa_output "${APP_DIR}/qa.jsonl" \
        --max "$MAX_QUESTIONS"; then
        log_error "Failed to execute question extraction script."
        return 1
    fi

    if [[ ! -s "$QUESTIONS_CSV" ]]; then
        log_error "Question file was not created or is empty: ${QUESTIONS_CSV}"
        return 1
    fi

    local question_count
    question_count=$(tail -n +2 "$QUESTIONS_CSV" | wc -l)
    log_success "Extraction complete. Extracted ${question_count} questions."
}

run_jmeter_test() {
    log_step "Running JMeter test plan..."

    cd "$APP_DIR"
    log_info "Starting JMeter with plan: $JMETER_PLAN"
    log_info "Results will be saved to: $RESULTS_FILE"
    log_info "Logs will be saved to: $LOG_FILE"

    if ! jmeter -n -t "$JMETER_PLAN" -l "$RESULTS_FILE" -j "$LOG_FILE"; then
        log_error "JMeter test run failed."
        [[ -f "$LOG_FILE" ]] && {
            log_error "Last 10 lines of JMeter log:"
            tail -10 "$LOG_FILE" >&2
        }
        return 1
    fi

    if [[ ! -s "$RESULTS_FILE" ]]; then
        log_error "JMeter results file was not created or is empty."
        return 1
    fi

    local result_count
    result_count=$(wc -l < "$RESULTS_FILE")
    log_success "JMeter test completed. Generated ${result_count} result lines."
}

display_results_summary() {
    log_step "Displaying results summary..."

    if [[ -f "$RESULTS_FILE" ]]; then
        log_info "Results file:   ${RESULTS_FILE}"
        log_info "File size:      $(du -h "$RESULTS_FILE" | cut -f1)"
        log_info "Total lines:    $(wc -l < "$RESULTS_FILE")"
    fi

    if [[ -f "$LOG_FILE" ]]; then
        log_info "Log file:       ${LOG_FILE}"
        log_info "File size:      $(du -h "$LOG_FILE" | cut -f1)"
    fi
}

# =============================================================================
# REPORT GENERATION
# =============================================================================

generate_report() {
    log_step "Generating JMeter HTML report..."
    # JMeter requires the output dir to be non-existent or empty
    rm -rf "$REPORT_DIR"
    if ! jmeter -g "$RESULTS_FILE" -o "$REPORT_DIR"; then
        log_error "Report generation failed."
        return 1
    fi
    log_success "HTML report generated at: ${REPORT_DIR}"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    cleanup_previous_outputs

    if ! verify_prerequisites; then
        exit 1
    fi

    if ! extract_questions; then
        exit 1
    fi

    if ! run_jmeter_test; then
        exit 1
    fi

    display_results_summary

    if ! generate_report; then
        exit 1
    fi

    log_success "All tasks completed successfully."
}

# Execute the main function
main
