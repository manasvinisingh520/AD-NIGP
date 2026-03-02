"""
Temporary script: load each embedding file from a directory, run PCA(2), and display
one scatter plot at a time. Close the figure window to see the next file.

Usage:
  python src/plot_embeddings_pca_temp.py
  python src/plot_embeddings_pca_temp.py --dir "e:/sf2026/GREmLN/embeddings"
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


EMBEDDINGS_DIR = "e:/sf2026/GREmLN/old_embeddings"
PATTERN = "*embeddings*.tsv"


def load_embedding_file(path: str):
    """Load embedding file: first column = IDs, rest = dimensions. Returns (X, ids)."""
    # GREmLN embedding files are comma-separated; first row is often a header (dim indices)
    try:
        df = pd.read_csv(path, sep=",", header=None)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] < 2:
        return None, None
    # Skip header row if first cell is NaN or numeric
    if pd.isna(df.iloc[0, 0]) or str(df.iloc[0, 0]).strip().lstrip("-").replace(".", "", 1).isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    if len(df) < 2:
        return None, None
    ids = df.iloc[:, 0].astype(str).tolist()
    X = df.iloc[:, 1:].astype(float).values
    # Drop rows with any NaN
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    ids = [ids[i] for i in range(len(ids)) if mask[i]]
    return X, ids


def main():
    parser = argparse.ArgumentParser(description="Plot PCA of each embedding file, one at a time.")
    parser.add_argument("--dir", type=str, default=EMBEDDINGS_DIR, help="Embeddings directory")
    parser.add_argument("--pattern", type=str, default=PATTERN, help="Glob pattern for files")
    args = parser.parse_args()

    dir_path = os.path.abspath(args.dir)
    if not os.path.isdir(dir_path):
        print(f"Not a directory: {dir_path}")
        return

    files = sorted(glob.glob(os.path.join(dir_path, args.pattern)))
    if not files:
        print(f"No files matching {args.pattern} in {dir_path}")
        return

    print(f"Found {len(files)} files. Close each figure to see the next.\n")

    for path in files:
        name = os.path.basename(path)
        print(f"Loading {name} ...")
        X, ids = load_embedding_file(path)
        if X is None or len(X) < 2:
            print(f"  Skip (no data or single row)\n")
            continue

        n_components = min(2, X.shape[1], X.shape[0] - 1)
        if n_components < 2:
            print(f"  Skip (need at least 2 dimensions or 2 samples)\n")
            continue

        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.5, s=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA: {name}\n(n={Z.shape[0]}, var explained: {pca.explained_variance_ratio_.sum():.2%})")
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        print(f"  Done.\n")

    print("Done all files.")


if __name__ == "__main__":
    main()
