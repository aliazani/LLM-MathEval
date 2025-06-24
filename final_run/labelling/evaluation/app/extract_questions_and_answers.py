#!/usr/bin/env python3
"""
extract_qa_csv_safe.py

1. Reads a JSONL file that contains entries like
   {"question": "...", "answer": "...#### the real answer"}
2. Produces
   * <questions_output>.csv   – one column called prompt
   * <qa_output>.jsonl       – question/ground-truth pairs

The CSV is written with:
• delimiter = '|'      (pick that in JMeter’s CSV Data Set Config)
• quoting   = QUOTE_NONE
• escapechar = '\\'    (so embedded |, " and \\ are back-slash-escaped)
The result is ready to drop into a JSON body like:
   {"prompt":"${prompt}"}
"""

import argparse
import csv
import json
from pathlib import Path


def json_escape(text: str) -> str:
    """
    Turn an arbitrary string into something that can sit safely inside double
    quotes in JSON, *without* being mangled by the CSV writer later on.

    Escapes:
      - backslashes (\\)
      - double-quotes (\")
      - newline and carriage return → space
    """
    return (
        text.replace('\\', '\\\\')
            .replace('"', r'\"')
            .replace('\r', ' ')
            .replace('\n', ' ')
            .strip()
    )


def extract_qa(input_path: Path, questions_output: Path,
               qa_output: Path, max_items: int) -> None:
    questions = []        # prompts (escaped only if needed)
    qa_pairs = []

    with input_path.open('r', encoding='utf-8') as fh:
        for idx, line in enumerate(fh):
            if idx >= max_items:
                break
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            q_raw = (entry.get('question') or '').strip()
            a_raw = (entry.get('answer') or '').strip()
            if not q_raw or not a_raw:
                continue

            # Extract ground truth after '####'
            ground_truth = a_raw.split('####')[-1].strip()

            # Only JSON-escape if the prompt contains problematic characters
            if any(ch in q_raw for ch in ('"', '\\', '\n', '\r')):
                q_safe = json_escape(q_raw)
            else:
                q_safe = q_raw

            questions.append(q_safe)
            qa_pairs.append({'question': q_raw, 'answer': ground_truth})

    # 1) write questions CSV (pipe-delimited, no quoting)
    with questions_output.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(
            csvfile,
            delimiter='|',
            quoting=csv.QUOTE_NONE,
            escapechar='\\',
            lineterminator='\n',
        )
        writer.writerow(['prompt'])
        for q in questions:
            writer.writerow([q])

    # 2) write question-answer JSONL
    with qa_output.open('w', encoding='utf-8') as qa_fh:
        for pair in qa_pairs:
            qa_fh.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"✅ Extracted {len(questions)} prompts  →  {questions_output}")
    print(f"✅ Extracted {len(qa_pairs)} QA pairs →  {qa_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract first-round questions and ground-truth answers into CSV (JSON-escaped if needed) and JSONL."
    )
    parser.add_argument('--input',            type=Path, required=True,
                        help="Path to input JSONL file")
    parser.add_argument('--questions_output', type=Path, required=True,
                        help="Output CSV file (prompts, escaped if needed)")
    parser.add_argument('--qa_output',        type=Path, required=True,
                        help="Output JSONL file with QA pairs")
    parser.add_argument('--max',              type=int, default=100,
                        help="Maximum number of entries to extract")
    args = parser.parse_args()

    extract_qa(args.input, args.questions_output, args.qa_output, args.max)
