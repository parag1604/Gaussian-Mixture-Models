"""
version0.py: the data, and why K-means is not enough.

What this file teaches
- The dataset has two clusters that share almost the same centre but are
  stretched in different directions (one along y = x, one along y = -x).
- K-means assigns each point to the nearest centre, so its clusters are always
  separated by straight lines (a Voronoi partition). It cannot use the *shape*
  of a cluster, and it fails here. Fixing this is the job of the Gaussian
  mixture model in version1.py.

This is the first file.
"""
import argparse

import numpy as np
from matplotlib import pyplot as plt

from utils import get_data, accuracy_up_to_permutation


def kmeans(X, k, iters=100, seed=0):
    # minimal K-means (Lloyd's algorithm), see the K-Means-Clustering repo
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((X[:, None] - centres) ** 2).sum(-1), axis=1)
        centres = np.array([X[labels == j].mean(0) for j in range(k)])
    return labels, centres


def main():
    parser = argparse.ArgumentParser(description="Visualize the data and run K-means on it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = get_data(seed=args.seed)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Ground truth")
    plt.savefig("data_vis.jpg")
    plt.close()

    labels, centres = kmeans(X, 2, seed=args.seed)
    print(f"K-means accuracy (best label matching): {accuracy_up_to_permutation(y, labels):.3f}")
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centres[:, 0], centres[:, 1], c="red", marker="x", s=100)
    plt.title("K-means (k = 2)")
    plt.savefig("kmeans_vis.jpg")
    print("Saved data_vis.jpg and kmeans_vis.jpg")


if __name__ == "__main__":
    main()
