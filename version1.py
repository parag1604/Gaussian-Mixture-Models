"""
version1.py: a Gaussian mixture model fitted with EM.

What is NEW compared to version0.py
- Each cluster is a full Gaussian (mean AND covariance) with a mixing weight,
  so clusters can be elongated and overlap.
- EM alternates soft assignments (E-step: responsibilities r_ij) with weighted
  maximum-likelihood updates (M-step). See GaussianMM.py.
- EM never decreases the log-likelihood; we plot it per iteration.
- EM is still a local method: a bad start can converge to a poor solution. Try
  a few --seed values.
"""
import argparse

from matplotlib import pyplot as plt

from utils import get_data, accuracy_up_to_permutation, plot_ellipses
from GaussianMM import GaussianMixtureModel


def main():
    parser = argparse.ArgumentParser(description="EM for a Gaussian mixture model")
    parser.add_argument("--k", type=int, default=2, help="number of components")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = get_data(seed=args.seed)

    gmm = GaussianMixtureModel(X, args.k, max_iter=args.max_iter, seed=args.seed)
    gmm.fit()
    y_pred = gmm.predict()

    print(f"EM stopped after {len(gmm.log_likelihoods)} iterations, "
          f"average log-likelihood = {gmm.log_likelihoods[-1]:.4f}")
    print("mixing weights:", gmm.pi.round(3))
    if args.k == 2:
        print(f"accuracy (best label matching): {accuracy_up_to_permutation(y, y_pred):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(X[:, 0], X[:, 1], c=y_pred, s=10)
    plot_ellipses(ax1, gmm.mu, gmm.sigma)
    ax1.set_title("GMM clusters with 1 and 2 std ellipses")
    ax2.plot(gmm.log_likelihoods, marker=".")
    ax2.set_xlabel("EM iteration")
    ax2.set_ylabel("average log-likelihood")
    ax2.set_title("EM never decreases the likelihood")
    plt.tight_layout()
    plt.savefig("pred_vis.jpg")
    print("Saved pred_vis.jpg")


if __name__ == "__main__":
    main()
