#!/usr/bin/env python3
import argparse
from pathlib import Path

from guidellm.utils.hf_transformers import check_load_processor  # import from guidellm

def download_tokenizer(model_name: str, save_dir: str):
    """Download a Hugging Face tokenizer using guidellm's check_load_processor."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load the tokenizer
    tokenizer = check_load_processor(
        processor=model_name,
        processor_args=None,
        error_msg=f"loading tokenizer for {model_name}"
    )

    # Save tokenizer to directory
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer for '{model_name}' saved to '{save_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a tokenizer using guidellm utils.")
    parser.add_argument("model_name", type=str, help="Name of the model on Hugging Face Hub")
    parser.add_argument("save_dir", type=str, help="Directory to save the tokenizer")
    args = parser.parse_args()

    download_tokenizer(args.model_name, args.save_dir)
