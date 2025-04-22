import torch
from umap import UMAP
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse  # Import argparse
from torchcfm import OTPlanSampler


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


def l2_norm_across_time(emb_dir, split):
    output_base = Path(emb_dir)
    emb_dir = output_base / "audio_emb"

    clean_path = emb_dir / split / "clean"
    noisy_path = emb_dir / split / "noisy"

    for clean_emb_file in sorted(list(clean_path.glob("*.pt"))):
        file_id = clean_emb_file.stem
        noisy_emb_file = noisy_path / f"{file_id}.pt"

        clean_emb = torch.load(clean_emb_file)
        noisy_emb = torch.load(noisy_emb_file)

        diff = torch.linalg.norm(clean_emb - noisy_emb, dim=2)[0].numpy()

        plt.plot(diff)
        plt.show()

        breakpoint()


def umap(emb_dir, split, aggregate="mean", N=100):
    output_base = Path(emb_dir)
    emb_dir = output_base / "audio_emb"

    clean_path = emb_dir / split / "clean"
    noisy_path = emb_dir / split / "noisy"

    clean_data = []
    noisy_data = []

    for clean_emb_file in sorted(list(clean_path.glob("*.pt")))[:N]:
        file_id = clean_emb_file.stem
        noisy_emb_file = noisy_path / f"{file_id}.pt"

        clean_emb = torch.load(clean_emb_file)
        noisy_emb = torch.load(noisy_emb_file)

        if aggregate == "mean":
            clean_data.append(clean_emb.mean(dim=0).numpy())
            noisy_data.append(noisy_emb.mean(dim=0).numpy())
        else:
            clean_data.append(clean_emb.numpy())
            noisy_data.append(noisy_emb.numpy())

    reducer = UMAP()
    if aggregate == "mean":
        umap_emb = reducer.fit_transform(
            np.concatenate(clean_data + noisy_data).reshape(N * 2, -1)
        )
    else:
        umap_emb = reducer.fit_transform(
            np.concatenate(clean_data + noisy_data).reshape(N * 2, -1)
        )
    plt.scatter(umap_emb[:N, 0], umap_emb[:N, 1], c="b", marker="o")
    plt.scatter(umap_emb[N:, 0], umap_emb[N:, 1], c="r", marker="+")

    for noisy_sample, clean_sample in zip(umap_emb[N:], umap_emb[:N]):
        d = clean_sample - noisy_sample
        plt.arrow(noisy_sample[0], noisy_sample[1], d[0], d[1], alpha=0.5)

    plt.show()

    ot_sampler = OTPlanSampler(method="exact")
    noisy_data_ot, clean_data_ot, label_x, label_y = ot_sampler.sample_plan_with_labels(
        torch.tensor(np.concatenate(noisy_data).reshape(N, -1)),
        torch.tensor(np.concatenate(clean_data).reshape(N, -1)),
        torch.arange(N),
        torch.arange(N),
        replace=False,
    )

    print(label_x)
    print(label_y)
    print(label_x == label_y)
    print((label_x == label_y).sum())
    # print(clean_data_ot - np.concatenate(clean_data)[label_y])
    # print(clean_data_ot == np.concatenate(clean_data)[label_y])
    # print((clean_data_ot == np.concatenate(clean_data)[label_y]).sum())

    if aggregate == "mean":
        umap_emb = reducer.fit_transform(
            torch.concatenate((clean_data_ot, noisy_data_ot)).numpy()
        )
    else:
        umap_emb = reducer.fit_transform(
            torch.concatenate((clean_data_ot, noisy_data_ot)).numpy().reshape(N * 2, -1)
        )
    plt.scatter(umap_emb[:N, 0], umap_emb[:N, 1], c="b", marker="o")
    plt.scatter(umap_emb[N:, 0], umap_emb[N:, 1], c="r", marker="+")
    for noisy_sample, clean_sample in zip(umap_emb[N:], umap_emb[:N]):
        d = clean_sample - noisy_sample
        plt.arrow(noisy_sample[0], noisy_sample[1], d[0], d[1], alpha=0.5)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare clean vs noisy Whisper embeddings."
    )

    parser.add_argument(
        "embedding_dir",
        type=str,
        help="Path to the root directory containing saved embeddings (e.g., ./VB+DMD_wspr). "
        "Expected structure: embedding_dir/audio_emb/{split}/{condition}/*.pt",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "valid"],
        help="Which data split to analyze (default: test).",
    )

    parser.add_argument(
        "--reducer",
        type=str.lower,
        default="umap",  # Convert input to lowercase
        choices=["umap", "tsne", "pca"],
        help="Dimensionality reduction technique for visualization (default: umap).",
    )

    parser.add_argument(
        "--no-plots", action="store_true", help="Disable displaying generated plots."
    )

    parser.add_argument(
        "--no-lines",
        action="store_true",
        help="Do not draw lines connecting clean/noisy pairs in the scatter plot.",
    )

    parser.add_argument(
        "--analysis",
        type=str.lower,
        default="l2_norm",  # Convert input to lowercase
        help="Type of analysis to compute",
    )
    parser.add_argument(
        "--N", type=int, default=100, help="Number of samples to analysis"
    )
    parser.add_argument(
        "--aggregate",
        type=str.lower,
        default="none",  # Convert input to lowercase
        help="Ways to aggragate samples",
    )

    args = parser.parse_args()

    if args.analysis == "l2_norm":
        l2_norm(
            emb_dir=args.embedding_dir,
            split=args.split,
        )
    elif args.analysis == "l2_norm_across_time":
        l2_norm_across_time(
            emb_dir=args.embedding_dir,
            split=args.split,
        )
    elif args.analysis == "umap":
        umap(
            emb_dir=args.embedding_dir,
            split=args.split,
            aggregate=args.aggregate,
            N=args.N,
        )


if __name__ == "__main__":
    main()
