"""
version2.py: numerically stable EM, restarts, and choosing k with BIC.

What is NEW compared to version1.py
- Log-domain E-step. Densities like N(x | mu, sigma) underflow to 0 in high
  dimensions or far from every component, and then r_ij = 0/0. Working with log
  densities and the log-sum-exp trick avoids this.
- Restarts: EM finds a local optimum, so we run it from several seeds and keep
  the run with the highest likelihood.
- Model selection: the likelihood always improves with more components, so it
  cannot choose k. The Bayesian information criterion
      BIC = -2 log L + (#parameters) log n
  penalizes the extra parameters; we pick the k with the lowest BIC.
"""
import argparse

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

from utils import get_data, accuracy_up_to_permutation, plot_ellipses
from GaussianMM import GaussianMixtureModel


class StableGaussianMixtureModel(GaussianMixtureModel):
    def _log_weighted_densities(self) -> np.ndarray:
        # log pi_j + log N(x_i | mu_j, sigma_j), never exponentiated
        return np.stack([np.log(self.pi[j]) + multivariate_normal(
            mean=self.mu[j], cov=self.sigma[j]).logpdf(self.X) for j in range(self.k)], axis=1)

    def e_step(self) -> np.ndarray:
        log_dens = self._log_weighted_densities()
        log_px = logsumexp(log_dens, axis=1, keepdims=True)   # log sum_j exp(.), computed stably
        self.log_likelihoods.append(log_px.mean())
        return np.exp(log_dens - log_px)

    def predict_proba(self, X: np.ndarray = None) -> np.ndarray:
        if X is not None:
            self.X, self.n = X, X.shape[0]
        log_dens = self._log_weighted_densities()
        return np.exp(log_dens - logsumexp(log_dens, axis=1, keepdims=True))

    def num_parameters(self) -> int:
        # (k - 1) weights + k means + k symmetric covariances
        return (self.k - 1) + self.k * self.d + self.k * self.d * (self.d + 1) // 2

    def bic(self) -> float:
        # log-likelihood of the data under the FINAL parameters
        total_log_likelihood = logsumexp(self._log_weighted_densities(), axis=1).sum()
        return -2 * total_log_likelihood + self.num_parameters() * np.log(self.n)


def fit_with_restarts(X, k, restarts, max_iter):
    best = None
    for seed in range(restarts):
        gmm = StableGaussianMixtureModel(X, k, max_iter=max_iter, seed=seed)
        gmm.fit()
        if best is None or gmm.log_likelihoods[-1] > best.log_likelihoods[-1]:
            best = gmm
    return best


def main():
    parser = argparse.ArgumentParser(description="Stable EM with restarts and BIC model selection")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="seed for the data")
    args = parser.parse_args()

    X, y = get_data(seed=args.seed)

    models, bics = {}, []
    for k in args.ks:
        models[k] = fit_with_restarts(X, k, args.restarts, args.max_iter)
        bics.append(models[k].bic())
        print(f"k = {k}: average log-likelihood = {models[k].log_likelihoods[-1]:.4f}, BIC = {bics[-1]:.1f}")

    best_k = args.ks[int(np.argmin(bics))]
    best = models[best_k]
    y_pred = best.predict()
    print(f"BIC chooses k = {best_k}")
    if best_k == 2:
        print(f"accuracy (best label matching): {accuracy_up_to_permutation(y, y_pred):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(args.ks, bics, marker="o")
    ax1.set_xlabel("number of components k")
    ax1.set_ylabel("BIC (lower is better)")
    ax1.set_title("Model selection")
    ax2.scatter(X[:, 0], X[:, 1], c=y_pred, s=10)
    plot_ellipses(ax2, best.mu, best.sigma)
    ax2.set_title(f"Best model by BIC (k = {best_k})")
    plt.tight_layout()
    plt.savefig("bic_vis.jpg")
    print("Saved bic_vis.jpg")


if __name__ == "__main__":
    main()
