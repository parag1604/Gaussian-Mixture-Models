import numpy as np
from typing import Tuple
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt


def get_data(
        num_points: int = 400,
        seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two crossing, elongated Gaussian clusters with almost the same centre.

    Distance-based methods (K-means) cannot separate them; the clusters differ
    in their *covariances*, which is exactly what a GMM models.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    # Cluster 1: positively correlated
    GaussianDistribution1 =\
            multivariate_normal(mean=[2, 3], cov=[[1, 0.9], [0.9, 1]])
    X.append(GaussianDistribution1.rvs(size=num_points//2, random_state=rng))
    y.append(np.zeros(num_points//2))

    # Cluster 2: negatively correlated
    GaussianDistribution2 =\
            multivariate_normal(mean=[2, 2], cov=[[1.2, -1.1], [-1.1, 1.2]])
    X.append(GaussianDistribution2.rvs(size=num_points//2, random_state=rng))
    y.append(np.ones(num_points//2))

    # Concatenate
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    return X, y


def accuracy_up_to_permutation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cluster ids are arbitrary, so score the best matching of ids to labels (2 clusters)."""
    acc = np.mean(y_true == y_pred)
    return max(acc, 1 - acc)


def plot_ellipses(ax, means, covs, color="red"):
    """Draw the 1- and 2-standard-deviation ellipses of each Gaussian."""
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)])
    for mu, cov in zip(means, covs):
        L = np.linalg.cholesky(cov)
        for r in (1, 2):
            pts = mu[:, None] + r * L @ circle
            ax.plot(pts[0], pts[1], color=color, lw=1)
