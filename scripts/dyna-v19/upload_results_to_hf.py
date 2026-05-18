#!/usr/bin/env python3
"""Upload current SVG-500 CEDI results to Hugging Face Icey444/tmp.

Uploads work_dirs/svg500/{v19d9,v19d91}-*_gpt54/ trees as a dataset repo.
Logs and the smoke dirs are excluded.

Usage:
    HF_TOKEN=... python scripts/dyna-v19/upload_results_to_hf.py [--repo-id Icey444/tmp]
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Icey444/tmp")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--root", default="work_dirs/svg500")
    ap.add_argument("--path-in-repo", default="cedi_svg500_gpt54")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN before running.")

    api = HfApi(token=token)
    try:
        create_repo(args.repo_id, repo_type=args.repo_type, token=token, exist_ok=True)
        print(f"Repo {args.repo_id} ({args.repo_type}) ready.")
    except Exception as e:
        print(f"create_repo warning: {e}")

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"Root {root} does not exist.")

    # Collect all v19d9-*_gpt54 and v19d91-*_gpt54 directories
    dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d.name.startswith("v19d9-") or d.name.startswith("v19d91-"))
    )
    if not dirs:
        sys.exit(f"No v19d9-* or v19d91-* dirs found in {root}.")

    print(f"Found {len(dirs)} variant dirs to upload:")
    for d in dirs:
        print(f"  - {d.name}")
    print()

    for d in dirs:
        target = f"{args.path_in_repo}/{d.name}"
        print(f"Uploading {d} -> {args.repo_id}:{target} ...")
        api.upload_folder(
            folder_path=str(d),
            path_in_repo=target,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Update {d.name} (in-progress snapshot)",
            allow_patterns=["*.json"],
            ignore_patterns=None,
        )
        print(f"  done.")

    print()
    print(f"All uploads complete.")
    print(f"Browse: https://huggingface.co/datasets/{args.repo_id}/tree/main/{args.path_in_repo}")


if __name__ == "__main__":
    main()
