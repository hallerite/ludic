#!/usr/bin/env python3
"""
Push a ludic checkpoint to HuggingFace Hub.

Usage:
  python scripts/push_to_hub.py checkpoints/step_000100 username/my-model
  python scripts/push_to_hub.py checkpoints/step_000100 username/my-model --private
  python scripts/push_to_hub.py checkpoints/step_000100 username/my-model --base-model Qwen/Qwen2.5-7B-Instruct

Assumes you are logged in via `huggingface-cli login`.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Push checkpoint to HuggingFace Hub")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    parser.add_argument("--commit-message", default="Upload model", help="Commit message")
    parser.add_argument("--base-model", help="Base model to copy tokenizer from (e.g., Qwen/Qwen2.5-7B-Instruct)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)

    # Upload checkpoint
    api.upload_folder(
        folder_path=str(ckpt_path),
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        ignore_patterns=["*.pt", "trainer_state.json"],
    )
    print(f"Pushed checkpoint to https://huggingface.co/{args.repo_id}")

    # Copy tokenizer from base model if specified
    if args.base_model:
        print(f"Copying tokenizer from {args.base_model}...")
        for filename in TOKENIZER_FILES:
            try:
                local_path = api.hf_hub_download(args.base_model, filename)
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=filename,
                    repo_id=args.repo_id,
                    commit_message=f"Add {filename} from {args.base_model}",
                )
                print(f"  Copied {filename}")
            except Exception:
                pass  # File doesn't exist in base model, skip


if __name__ == "__main__":
    main()
