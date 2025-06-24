#!/usr/bin/env python3

import csv
import json
import os
import requests
import re
from typing import Dict, List, Optional

class AnswerEvaluator:
    def __init__(self, ollama_host="localhost", ollama_port=11434):
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.base_url = f"http://{ollama_host}:{ollama_port}"
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"

    def check_ollama_status(self):
        """Check if Ollama is running and list available models."""
        try:
            print(f"Checking Ollama status at {self.base_url}...")
            response = requests.get(self.tags_url, timeout=10)
            response.raise_for_status()
            models = response.json()

            print(f"✓ Ollama is running")
            print(f"Available models: {[model['name'] for model in models.get('models', [])]}")

            # Check if llama3 is available
            model_names = [model['name'] for model in models.get('models', [])]
            llama3_available = any('llama3' in name.lower() for name in model_names)

            if not llama3_available:
                print("⚠ Warning: No Llama3 model found. Available models:")
                for model in model_names:
                    print(f"  - {model}")
                print("\nTo install Llama3, run: ollama pull llama3")
                return False

            return True

        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to Ollama at {self.base_url}")
            print("Make sure Ollama is running with: ollama serve")
            return False
        except Exception as e:
            print(f"✗ Error checking Ollama status: {e}")
            return False

    def get_available_llama_model(self):
        """Get the first available Llama3 model."""
        try:
            response = requests.get(self.tags_url, timeout=10)
            response.raise_for_status()
            models = response.json()

            model_names = [model['name'] for model in models.get('models', [])]

            # Look for llama3 variants
            for name in model_names:
                if 'llama3' in name.lower():
                    return name

            # Fallback to any available model
            if model_names:
                print(f"Warning: No Llama3 found, using {model_names[0]}")
                return model_names[0]

            return None

        except Exception as e:
            print(f"Error getting models: {e}")
            return None

    def extract_answer_with_llama(self, generated_text: str) -> str:
        """Use Llama to extract the final numerical answer from generated text."""

        # Use the same prompt style as your working judge.py
        prompt = (
            "You will be given a model's reasoning and answer.\n"
            "Extract the final numeric answer and reply with ONLY that number.\n"
            "If none is found, reply with the single word: NONE.\n\n"
            f"{generated_text}\n\n###\nNumber:"
        )

        # Try llama3.3 first (from your judge.py), then fallback to available models
        model_name = "llama3.3"
        available_model = self.get_available_llama_model()

        if not available_model:
            print("No models available")
            return "ERROR"

        # Use the exact same payload structure as your judge.py
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 4,  # Using max_tokens like in your judge.py
            "stream": False,
        }

        try:
            print(f"Calling Ollama with model: {model_name}")
            response = requests.post(self.generate_url, json=payload, timeout=60)

            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"Response text: {response.text}")
                # If llama3.3 fails, try the available model
                if model_name != available_model:
                    print(f"Retrying with available model: {available_model}")
                    payload["model"] = available_model
                    response = requests.post(self.generate_url, json=payload, timeout=60)
                    response.raise_for_status()
                else:
                    response.raise_for_status()

            result = response.json()

            # Use the same response parsing as your judge.py
            extracted = (result.get("response") or
                        result.get("choices", [{}])[0].get("text", "")).strip()

            print(f"Raw Llama response: '{extracted}'")

            # Return None if model says NONE, otherwise return the extracted text
            if extracted.upper() == "NONE":
                return "NONE"

            # Try to extract just the number using regex as backup
            number_match = re.search(r'\b\d+(?:\.\d+)?\b', extracted)
            if number_match:
                return number_match.group()

            return extracted

        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}")
            print(f"Make sure Ollama is running and accessible at {self.base_url}")
            return "ERROR"
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            return "ERROR"
        except Exception as e:
            print(f"Error calling Llama: {e}")
            return "ERROR"

    def load_ground_truth(self, jsonl_path: str) -> Dict[str, str]:
        """Load ground truth Q&A pairs from JSONL file."""
        qa_pairs = {}

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        qa_pairs[data['question']] = data['answer']
        except Exception as e:
            print(f"Error loading ground truth: {e}")

        return qa_pairs

    def load_csv_data(self, csv_path: str) -> List[Dict]:
        """Load CSV data with LLM generations."""
        csv_data = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('attributes.input.value') and row.get('attributes.output.value'):
                        csv_data.append({
                            'question': row['attributes.input.value'].strip(),
                            'generated_answer': row['attributes.output.value'].strip(),
                            'generation_time': row.get('attributes.generation_time_sec', 'N/A'),
                            'original_row': row  # Keep original row data for output
                        })
        except Exception as e:
            print(f"Error loading CSV: {e}")

        return csv_data

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer or answer in ["ERROR", "NONE"]:
            return answer

        # Remove any non-digit characters except decimal point
        normalized = re.sub(r'[^\d.]', '', str(answer))

        # Handle decimal numbers
        try:
            if '.' in normalized:
                return str(float(normalized))
            else:
                return str(int(normalized))
        except ValueError:
            return str(answer).strip()

    def write_results_csv(self, results_data: List[Dict], output_path: str):
        """Write evaluation results to CSV file with correctness field."""
        if not results_data:
            print("No results to write")
            return

        # Define the field order (original fields + correctness)
        original_fields = [
            'name', 'span_kind', 'parent_id', 'start_time', 'end_time',
            'status_code', 'status_message', 'events', 'context.span_id',
            'context.trace_id', 'attributes.output.value', 'attributes.llm',
            'attributes.generation_time_sec', 'attributes.input.value'
        ]

        # Get all fields from first row and add correctness
        all_fields = list(results_data[0].keys())

        # Ensure original fields come first, then add any extra fields, then correctness
        fieldnames = []
        for field in original_fields:
            if field in all_fields:
                fieldnames.append(field)

        # Add any additional fields that weren't in the original list
        for field in all_fields:
            if field not in fieldnames and field not in ['question', 'generated_answer', 'extracted_answer', 'ground_truth', 'correctness', 'original_row']:
                fieldnames.append(field)

        # Add correctness as the last field
        fieldnames.append('correctness')

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for row in results_data:
                    # Only write the fields we want (exclude helper fields)
                    output_row = {field: row.get(field, '') for field in fieldnames}
                    writer.writerow(output_row)

            print(f"✓ Results saved to: {output_path}")

        except Exception as e:
            print(f"Error writing results CSV: {e}")

    def write_summary_file(self, total_count: int, correct_count: int, accuracy: float, 
                          duplicate_count: int, unique_count: int, output_path: str):
        """Write summary statistics to file."""
        try:
            summary_data = {
                'total_questions_processed': unique_count,
                'total_rows_in_csv': total_count + duplicate_count,
                'duplicate_questions_skipped': duplicate_count,
                'correct_answers': correct_count,
                'incorrect_answers': unique_count - correct_count,
                'accuracy_percentage': round(accuracy, 2)
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)

            print(f"✓ Summary saved to: {output_path}")

        except Exception as e:
            print(f"Error writing summary file: {e}")

    def evaluate_answers(self, csv_path: str, jsonl_path: str, output_csv: str = None, summary_file: str = None):
        """Main evaluation function with duplicate detection."""

        # Check if Ollama is running first
        if not self.check_ollama_status():
            print("\nCannot proceed without Ollama. Please:")
            print("1. Start Ollama server: ollama serve")
            print("2. Install Llama3 model: ollama pull llama3")
            print("3. Check host/port configuration")
            return

        print("\nLoading data...")
        ground_truth = self.load_ground_truth(jsonl_path)
        csv_data = self.load_csv_data(csv_path)

        print(f"Loaded {len(ground_truth)} ground truth pairs")
        print(f"Loaded {len(csv_data)} generated responses")
        print("=" * 80)

        # Set to track processed questions and avoid duplicates
        processed_questions = set()
        results_data = []
        
        correct_count = 0
        total_count = 0
        duplicate_count = 0

        for i, data in enumerate(csv_data, 1):
            question = data['question']
            generated_answer = data['generated_answer']
            original_row = data['original_row']

            # Check for duplicates
            if question in processed_questions:
                duplicate_count += 1
                print(f"\n[{i}] Skipping duplicate question (#{duplicate_count})")
                print(f"Question: {question[:100]}...")
                
                # Still add to results but mark as duplicate
                result_row = original_row.copy()
                result_row['correctness'] = 'DUPLICATE'
                results_data.append(result_row)
                continue

            # Add question to processed set
            processed_questions.add(question)

            # Find matching ground truth
            gt_answer = ground_truth.get(question)

            if gt_answer is None:
                print(f"\n[{i}] No ground truth found for question")
                print(f"Question: {question[:100]}...")
                
                # Add to results but mark as no ground truth
                result_row = original_row.copy()
                result_row['correctness'] = 'NO_GROUND_TRUTH'
                results_data.append(result_row)
                continue

            print(f"\n[{i}] Processing unique question #{len(processed_questions)}...")
            print(f"Question: {question}")
            print(f"\nLLM Generated Answer:")
            print(f"{generated_answer}")

            # Extract answer using Llama3
            print(f"\nExtracting final answer with Llama3...")
            extracted_answer = self.extract_answer_with_llama(generated_answer)

            # Normalize answers for comparison
            normalized_extracted = self.normalize_answer(extracted_answer)
            normalized_gt = self.normalize_answer(gt_answer)

            # Check if correct
            is_correct = normalized_extracted == normalized_gt

            print(f"\nExtracted Answer: {extracted_answer}")
            print(f"Ground Truth: {gt_answer}")
            print(f"Correct: {'✓' if is_correct else '✗'}")

            # Add to results with correctness
            result_row = original_row.copy()
            result_row['correctness'] = 'true' if is_correct else 'false'
            results_data.append(result_row)

            if is_correct:
                correct_count += 1
            total_count += 1

            print("-" * 80)

        # Calculate final statistics
        unique_count = len(processed_questions)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS:")
        print(f"Total CSV Rows: {len(csv_data)}")
        print(f"Duplicate Questions Skipped: {duplicate_count}")
        print(f"Unique Questions Processed: {unique_count}")
        print(f"Questions with Ground Truth: {total_count}")
        print(f"Correct Answers: {correct_count}")
        print(f"Incorrect Answers: {total_count - correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*80}")

        # Write results to files
        if output_csv is None:
            output_csv = "/app/output/results.csv"
        if summary_file is None:
            summary_file = "/app/output/summary.json"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)

        self.write_results_csv(results_data, output_csv)
        self.write_summary_file(total_count, correct_count, accuracy, duplicate_count, unique_count, summary_file)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate LLM answers using Llama3 extraction')
    parser.add_argument('--csv', required=True, help='Path to CSV file with LLM generations')
    parser.add_argument('--jsonl', required=True, help='Path to JSONL file with ground truth')
    parser.add_argument('--host', default='localhost', help='Ollama host (default: localhost)')
    parser.add_argument('--port', default=11434, type=int, help='Ollama port (default: 11434)')
    parser.add_argument('--output-csv', help='Path for output CSV file (default: auto-generated)')
    parser.add_argument('--summary-file', help='Path for summary JSON file (default: auto-generated)')

    args = parser.parse_args()

    evaluator = AnswerEvaluator(ollama_host=args.host, ollama_port=args.port)
    evaluator.evaluate_answers(args.csv, args.jsonl, args.output_csv, args.summary_file)

if __name__ == "__main__":
    main()
