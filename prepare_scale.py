import os
import sys
import pickle
import argparse
import numpy as np
import tiktoken
from pathlib import Path

SUPPORTED_EXTENSIONS = {".py", ".sh", ".yaml", ".yml", ".json", ".rb", ".lua", ".txt"}
BPE_ENCODING = "cl100k_base"  

def collect_source_files(raw_repos_dir: Path) -> list[Path]:
    """Recursively collect all supported source files under raw_repos/."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(raw_repos_dir.rglob(f"*{ext}"))
    files.sort()   # deterministic ordering
    print(f"[prepare] Found {len(files)} source files in {raw_repos_dir}")
    return files

def encode_files(files: list[Path], enc: tiktoken.Encoding) -> list[int]:
    """
    Read + encode every file, concatenating all token IDs into a single list.
    A special <|endoftext|> boundary token (id=100257 in cl100k_base) is
    inserted between documents so the model learns document boundaries.
    """
    all_tokens: list[int] = []
    eot = enc.eot_token        

    for i, fpath in enumerate(files):
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"  [WARN] Skipping {fpath}: {exc}", file=sys.stderr)
            continue

        tokens = enc.encode_ordinary(text)   
        all_tokens.extend(tokens)
        all_tokens.append(eot)             

        if (i + 1) % 500 == 0:
            print(f"  [prepare] Encoded {i+1}/{len(files)} files …  total tokens so far: {len(all_tokens):,}")

    return all_tokens

def write_bin(path: Path, tokens: list[int]) -> None:
    """Persist token IDs as a memory-mapped uint32 numpy array (.bin file)."""
    arr = np.array(tokens, dtype=np.uint32)
    arr.tofile(str(path))
    print(f"  [prepare] Wrote {len(arr):,} tokens → {path}  ({path.stat().st_size / 1e6:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(description="SecCodeGPT BPE Data Preparation")
    parser.add_argument("--data_dir", default="data/seccode",
                        help="Root data directory (must contain raw_repos/)")
    parser.add_argument("--split", type=float, default=0.9,
                        help="Fraction of data used for training (default: 0.90)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    raw_dir    = data_dir / "raw_repos"
    train_path = data_dir / "train.bin"
    val_path   = data_dir / "val.bin"
    meta_path  = data_dir / "meta.pkl"

    if not raw_dir.exists():
        sys.exit(f"[ERROR] raw_repos directory not found: {raw_dir}")

    # 1. Load BPE tokenizer
    print(f"[prepare] Loading tiktoken encoding: {BPE_ENCODING}")
    enc = tiktoken.get_encoding(BPE_ENCODING)
    vocab_size = enc.n_vocab
    print(f"[prepare] Vocabulary size: {vocab_size:,}")

    # 2. Collect and  encode
    source_files = collect_source_files(raw_dir)
    if not source_files:
        sys.exit("[ERROR] No source files found. Check SUPPORTED_EXTENSIONS or raw_repos/ contents.")

    all_tokens = encode_files(source_files, enc)
    total = len(all_tokens)
    print(f"\n[prepare] Total tokens after encoding: {total:,}")

    # 3. Train / val split
    n_train = int(total * args.split)
    train_tokens = all_tokens[:n_train]
    val_tokens   = all_tokens[n_train:]
    print(f"[prepare] Train tokens: {len(train_tokens):,}  |  Val tokens: {len(val_tokens):,}")

    # 4. Write .bin files
    write_bin(train_path, train_tokens)
    write_bin(val_path,   val_tokens)

    # 5. Save metadata
    meta = {
        "vocab_size":    vocab_size,
        "encoding_name": BPE_ENCODING,
        "eot_token":     enc.eot_token,
        "n_train":       len(train_tokens),
        "n_val":         len(val_tokens),
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[prepare] Saved metadata → {meta_path}")
    print("[prepare] Done ✓")


if __name__ == "__main__":
    main()