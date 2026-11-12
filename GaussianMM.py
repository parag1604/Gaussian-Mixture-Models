import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    """Gaussian mixture model fitted with Expectation-Maximization (EM).

    Parameters: mixing weights pi_j, means mu_j and covariances sigma_j.
    E-step: responsibilities r_ij = p(z_i = j | x_i) under the current parameters.
    M-step: weighted maximum-likelihood updates, using r_ij as soft counts.
    """

    def __init__(
            self,
            X: np.ndarray,
            k: int = 2,
            max_iter: int = 100,
            tol: float = 1e-6,
            reg: float = 1e-6,
            seed: int = None,
    ) -> None:
        self.X = X
        self.k = k
        self.max_iter = max_iter
        self.tol = tol      # stop when the average log-likelihood improves less than this
        self.reg = reg      # added to the covariance diagonals so they stay invertible
        self.n, self.d = X.shape
        rng = np.random.default_rng(seed)
        # Means start at k random data points. They must be FLOAT arrays: an
        # integer array would silently truncate every M-step update.
        self.mu = X[rng.choice(self.n, self.k, replace=False)].astype(float)
        self.sigma = np.array([np.cov(X.T) for _ in range(self.k)])  # start broad
        self.pi = np.ones(self.k) / self.k                            # mixing weights
        self.log_likelihoods = []

    def _weighted_densities(self) -> np.ndarray:
        # (n, k) matrix with entries pi_j * N(x_i | mu_j, sigma_j)
        dens = np.zeros((self.n, self.k))
        for j in range(self.k):
            dens[:, j] = self.pi[j] * multivariate_normal(
                mean=self.mu[j], cov=self.sigma[j]).pdf(self.X)
        return dens

    def e_step(self) -> np.ndarray:
        dens = self._weighted_densities()
        self.log_likelihoods.append(np.mean(np.log(dens.sum(axis=1))))
        # Bayes' rule: posterior over the component that generated each point
        return dens / dens.sum(axis=1, keepdims=True)

    def m_step(self, posteriors: np.ndarray) -> None:
        for j in range(self.k):
            r = posteriors[:, j]
            N_j = r.sum()                         # effective number of points in component j
            self.pi[j] = N_j / self.n
            self.mu[j] = (r[:, None] * self.X).sum(axis=0) / N_j
            diff = self.X - self.mu[j]
            self.sigma[j] = (r[:, None] * diff).T @ diff / N_j + self.reg * np.eye(self.d)

    def fit(self) -> None:
        for _ in range(self.max_iter):
            posteriors = self.e_step()
            self.m_step(posteriors)
            # EM never decreases the likelihood; stop once it has flattened out
            if len(self.log_likelihoods) > 1 and \
                    self.log_likelihoods[-1] - self.log_likelihoods[-2] < self.tol:
                break

    def predict_proba(self, X: np.ndarray = None) -> np.ndarray:
        if X is not None:
            self.X, self.n = X, X.shape[0]
        dens = self._weighted_densities()   # uses the FINAL parameters
        return dens / dens.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        # most probable component, including the mixing weight pi_j (the prior)
        return np.argmax(self.predict_proba(X), axis=1)
