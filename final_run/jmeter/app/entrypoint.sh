#!/usr/bin/env bash

# =============================================================================
# SCRIPT CONFIGURATION
#
# This script automates the process of setting up and running a JMeter
# experiment three times, generating an HTML report after each run, and
# sleeping 5 minutes between each run.
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

readonly MAX_QUESTIONS=1500

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
}

log_info() {
    log "    $*"
}

log_success() {
    log "✅  $*"
}

log_error() {
    log "❌  $*" >&2
}

log_step() {
    log "🔄  $*"
}

cleanup_files() {
    for file in "$@"; do
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

    cleanup_files \
      "${APP_DIR}/jmeter.log" \
      "${APP_DIR}/results_run1.jtl" "${APP_DIR}/jmeter_run1.log" \
      "${APP_DIR}/results_run2.jtl" "${APP_DIR}/jmeter_run2.log" \
      "${APP_DIR}/results_run3.jtl" "${APP_DIR}/jmeter_run3.log"

    # Remove any old report directories
    rm -rf "${APP_DIR}/report_run1" "${APP_DIR}/report_run2" "${APP_DIR}/report_run3"

    log_success "Cleanup complete."
}

verify_prerequisites() {
    log_step "Verifying prerequisites..."

    local cmds=(git python3 jmeter curl unzip nc)
    for cmd in "${cmds[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command not found: '$cmd'"
            return 1
        fi
    done

    if [[ ! -d "$MATHCHAT_REPO" ]]; then
        log_info "Cloning MathChat repo..."
        git clone --depth 1 "$MATHCHAT_REPO_URL" "$MATHCHAT_REPO" \
          || { log_error "Clone failed"; return 1; }
        log_success "Repository cloned."
    else
        log_info "MathChat repo already present."
    fi

    for file in "$FOLLOW_UP_JSON" "$EXTRACT_SCRIPT" "$JMETER_PLAN"; do
        if [[ ! -f "$file" ]]; then
            log_error "Missing required file: $file"
            return 1
        fi
    done

    log_success "All prerequisites satisfied."
}

extract_questions() {
    log_step "Extracting questions..."

    if [[ -f "$QUESTIONS_CSV" && "$FOLLOW_UP_JSON" -ot "$QUESTIONS_CSV" ]]; then
        log_info "Questions CSV up-to-date."
        return 0
    fi

    rm -f "$QUESTIONS_CSV"
    python3 "$EXTRACT_SCRIPT" \
        --input "$FOLLOW_UP_JSON" \
        --questions_output "$QUESTIONS_CSV" \
        --qa_output "${APP_DIR}/qa.jsonl" \
        --max "$MAX_QUESTIONS" \
      || { log_error "Extraction script failed"; return 1; }

    [[ -s "$QUESTIONS_CSV" ]] || { log_error "Questions CSV is empty"; return 1; }

    local count
    count=$(tail -n +2 "$QUESTIONS_CSV" | wc -l)
    log_success "Extracted ${count} questions."
}

run_jmeter_test() {
    log_step "Running JMeter plan..."

    cd "$APP_DIR"
    log_info "Plan:    $JMETER_PLAN"
    log_info "Results: $RESULTS_FILE"
    log_info "Log:     $LOG_FILE"

    jmeter -n -t "$JMETER_PLAN" -l "$RESULTS_FILE" -j "$LOG_FILE" \
      || { log_error "JMeter failed"; return 1; }

    [[ -s "$RESULTS_FILE" ]] || { log_error "Results file empty"; return 1; }

    local lines
    lines=$(wc -l < "$RESULTS_FILE")
    log_success "Run complete: ${lines} lines in results."
}

display_results_summary() {
    log_step "Results summary for ${RESULTS_FILE}:"
    log_info "Size:  $(du -h "$RESULTS_FILE" | cut -f1)"
    log_info "Lines: $(wc -l < "$RESULTS_FILE")"
}

display_log_summary() {
    log_step "Log summary for ${LOG_FILE}:"
    log_info "Size:  $(du -h "$LOG_FILE" | cut -f1)"
    log_info "Last 10 lines of log:"
    tail -n 10 "$LOG_FILE" | sed 's/^/    /'
}

# =============================================================================
# REPORT GENERATION
# =============================================================================

generate_report() {
    log_step "Generating JMeter HTML report for run ${run_num}..."
    local report_dir="${APP_DIR}/report_run${run_num}"
    rm -rf "$report_dir"
    if ! jmeter -g "$RESULTS_FILE" -o "$report_dir"; then
        log_error "Report generation failed for run ${run_num}."
        return 1
    fi
    log_success "HTML report for run ${run_num} generated at: ${report_dir}"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    cleanup_previous_outputs

    verify_prerequisites
    extract_questions

    for run_num in 1 2 3; do
        RESULTS_FILE="${APP_DIR}/results_run${run_num}.jtl"
        LOG_FILE="${APP_DIR}/jmeter_run${run_num}.log"

        log_step "=== Starting JMeter run #${run_num} ==="
        run_jmeter_test
        display_results_summary
        display_log_summary

        generate_report

        log_success "Run ${run_num} completed and report generated."

        if (( run_num < 3 )); then
            log_step "Sleeping for 5 minutes before next run..."
            sleep 300
        fi
    done

    log_success "All 3 runs and reports completed."
}

main
