import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse # Import argparse

def l2_norm(emb_dir, split):
    output_base = Path(emb_dir)
    emb_dir = output_base / "audio_emb"

    clean_path = emb_dir / split / "clean"
    noisy_path = emb_dir / split / "noisy"

    for clean_emb_file in sorted(list(clean_path.glob("*.pt"))):
        file_id = clean_emb_file.stem
        noisy_emb_file = noisy_path / f"{file_id}.pt"

        clean_emb = torch.load(clean_emb_file)
        noisy_emb = torch.load(noisy_emb_file)

        print(torch.linalg.norm(clean_emb - noisy_emb))

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare clean vs noisy Whisper embeddings.")

    parser.add_argument("embedding_dir", type=str,
                        help="Path to the root directory containing saved embeddings (e.g., ./VB+DMD_wspr). "
                             "Expected structure: embedding_dir/audio_emb/{split}/{condition}/*.pt")

    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "valid"],
                        help="Which data split to analyze (default: test).")

    parser.add_argument("--reducer", type=str.lower, default="umap", # Convert input to lowercase
                        choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction technique for visualization (default: umap).")

    parser.add_argument("--no-plots", action="store_true",
                        help="Disable displaying generated plots.")

    parser.add_argument("--no-lines", action="store_true",
                        help="Do not draw lines connecting clean/noisy pairs in the scatter plot.")

    parser.add_argument("--analysis", type=str.lower, default="l2_norm", # Convert input to lowercase
                        help="Type of analysis to compute")

    args = parser.parse_args()

    if args.analysis == "l2_norm":
        l2_norm(
            emb_dir=args.embedding_dir,
            split=args.split,
        )

if __name__ == "__main__":
    main()
