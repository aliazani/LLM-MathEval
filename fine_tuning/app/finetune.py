#!/usr/bin/env python3
"""
Fine-tuning script for Mistral 7B Instruct with LoRA mathchat adapter.
Features:
- CLI args for model, data, hyperparameters
- Training/validation split and metrics logging
- Multiprocessing for dataset tokenization
- Reproducibility (seed)
- Progress bars and logging
- Push model/tokenizer to hub
"""
import os
import argparse
import logging
import torch
import gdown
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login


def download_data(dataset_url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logging.info(f"Downloading dataset to {out_path}")
    gdown.download(id=dataset_url, output=out_path, quiet=False)


def prepare_dataset(json_path: str, split_ratio: float, seed: int):
    raw = load_dataset("json", data_files=json_path, split="train")
    def to_chat_format(ex):
        msgs = [
            {"role": "user" if t["from"] == "human" else "assistant", "content": t["value"]}
            for t in ex["conversations"]
        ]
        return {"messages": msgs}
    ds = raw.map(to_chat_format, remove_columns=raw.column_names)
    return ds.train_test_split(test_size=split_ratio, seed=seed)


def load_model_and_tokenizer(model_name: str, tokenizer_name: str, trust_remote_code: bool):
    logging.info(f"Loading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
    tok.pad_token = tok.eos_token

    logging.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return model, tok


def apply_lora(model, r: int, alpha: int, dropout: float):
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none"
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def tokenize_fn(example, tokenizer, max_len: int):
    prompt = tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)
    enc = tokenizer(prompt, padding="max_length", truncation=True, max_length=max_len)
    enc["labels"] = enc["input_ids"].copy()
    return enc


def main():
    parser = argparse.ArgumentParser("Fine-tune Mistral 7B with LoRA")
    parser.add_argument("--dataset_url", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/mathchat.json")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--tokenizer_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output_dir", type=str, default="aliazn/mathchat-mistral")
    parser.add_argument("--split_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    if args.hf_token:
        login(args.hf_token)
        logging.info("Logged in to Hugging Face Hub.")
    else:
        logging.warning("No HF_TOKEN provided; pushing to hub will fail.")

    download_data(args.dataset_url, args.dataset_path)
    ds_splits = prepare_dataset(args.dataset_path, args.split_ratio, args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.tokenizer_name, args.trust_remote_code
    )
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # Tokenize with multiprocessing
    tokenize = lambda ex: tokenize_fn(ex, tokenizer, args.max_length)
    ds_tokenized = ds_splits.map(
        tokenize,
        batched=False,
        remove_columns=["messages"],
        num_proc=os.cpu_count(),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=50,
        eval_strategy="epoch",              
        save_strategy="epoch",
        fp16=True,
        seed=args.seed,
        push_to_hub=True,
        hub_model_id=args.output_dir,
        report_to="none",
        label_names=["labels"],              # ensure Trainer knows where to find your targets
        remove_unused_columns=False,         # prevent dropping of your labels column
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.push_to_hub(commit_message="Fine-tuned with LoRA mathchat adapter")


if __name__ == "__main__":
    main()

