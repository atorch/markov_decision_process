"""Hidden Markov model with Gaussian emissions, for asset returns that switch between regimes.

A Python port (generalized to a vector of returns) of the Gaussian regime-switching HMM in
https://github.com/atorch/finance_playground/pull/1 (hmm_gaussian_functions.R), which follows sec. 6.3 of
Ramon van Handel's HMM notes (https://www.princeton.edu/~rvan/orf557/hmm080728.pdf, algorithms 3.2, 3.4 and 6.1):

    X_k is a discrete hidden regime in {0, ..., K - 1}, a Markov chain with transition matrix P
    Y_k = m(X_k) + chol(S(X_k)) @ eta_k,    eta_k iid N(0, I)

where Y_k is a vector of log returns (one element per asset), m(i) is regime i's vector of mean log returns,
and S(i) is regime i's covariance matrix of log returns. With a single asset this is exactly the R model.

Notation: P[i, j] = Pr(X_{k+1} = j | X_k = i), and a belief is a probability vector over the K regimes.
"""

from dataclasses import dataclass, field

import numpy as np

# Floor for probabilities and densities before taking logs or dividing
TINY = 1e-300


@dataclass
class GaussianHMM:

    transition_matrix: np.ndarray  # K x K, rows sum to 1
    means: np.ndarray  # K x N: mean log return of each asset in each regime
    covariances: np.ndarray  # K x N x N: covariance matrix of log returns in each regime
    initial_distribution: np.ndarray | None = None  # K, defaults to the stationary distribution
    regime_names: tuple[str, ...] | None = None
    loglik_path: list = field(default_factory=list)  # Filled in by fit_gaussian_hmm

    def __post_init__(self):
        self.transition_matrix = np.atleast_2d(np.asarray(self.transition_matrix, dtype=float))
        self.means = np.atleast_2d(np.asarray(self.means, dtype=float))
        self.covariances = np.asarray(self.covariances, dtype=float).reshape(
            (self.n_regimes, self.n_assets, self.n_assets)
        )
        if self.initial_distribution is None:
            self.initial_distribution = stationary_distribution(self.transition_matrix)
        self.initial_distribution = np.asarray(self.initial_distribution, dtype=float)
        if self.regime_names is None:
            self.regime_names = tuple(f"regime {i}" for i in range(self.n_regimes))
        self.validate()

    @property
    def n_regimes(self):
        return self.transition_matrix.shape[0]

    @property
    def n_assets(self):
        return self.means.shape[1]

    def validate(self):
        K, N = self.n_regimes, self.n_assets
        assert self.transition_matrix.shape == (K, K)
        assert np.all(self.transition_matrix >= 0)
        assert np.allclose(self.transition_matrix.sum(axis=1), 1.0)
        assert self.means.shape == (K, N)
        assert self.covariances.shape == (K, N, N)
        assert np.allclose(self.covariances, np.swapaxes(self.covariances, 1, 2))
        assert self.initial_distribution.shape == (K,)
        assert np.isclose(self.initial_distribution.sum(), 1.0)
        assert len(self.regime_names) == K

    def informative_assets(self):
        # Assets whose returns carry information about the regime. An asset with zero variance in every regime and
        # the same mean in every regime (a bank account paying a fixed interest rate) is dropped from the likelihood:
        # its density is a point mass, and it says nothing about the regime anyway
        variances = np.diagonal(self.covariances, axis1=1, axis2=2)
        degenerate = np.all(variances == 0, axis=0)
        assert np.all(np.ptp(self.means[:, degenerate], axis=0) == 0), "A riskless asset's return can't depend on the regime"
        assert np.all(variances[:, ~degenerate] > 0), "An asset must be risky in every regime, or riskless in every regime"

        return np.flatnonzero(~degenerate)

    def log_densities(self, log_returns):
        # log N(log_returns; m(i), S(i)) for every regime i. Vectorized: log_returns has shape (..., N),
        # and the result has shape (..., K)
        idx = self.informative_assets()
        y = np.asarray(log_returns, dtype=float)[..., idx]
        means = self.means[:, idx]
        covariances = self.covariances[np.ix_(np.arange(self.n_regimes), idx, idx)]

        deviations = y[..., np.newaxis, :] - means  # (..., K, n)
        cholesky = np.linalg.cholesky(covariances)  # (K, n, n)
        whitened = np.linalg.solve(cholesky, deviations[..., np.newaxis])[..., 0]  # (..., K, n)
        log_det = 2 * np.sum(np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=-1)  # (K,)

        return -0.5 * (np.sum(whitened**2, axis=-1) + log_det + idx.size * np.log(2 * np.pi))

    def predict(self, belief):
        # Pr(X_{k+1} | Y_1, ..., Y_k) from Pr(X_k | Y_1, ..., Y_k): one step of the Markov chain
        return np.asarray(belief) @ self.transition_matrix

    def update(self, belief, log_returns):
        # One step of the forward algorithm (van Handel algorithm 3.2): predict, then condition on the new returns
        # Vectorized: belief has shape (..., K) and log_returns has shape (..., N)
        # Returns the new belief Pr(X_{k+1} | Y_1, ..., Y_{k+1}) and the predictive density Pr(Y_{k+1} | Y_1, ..., Y_k)
        predicted = self.predict(belief)
        log_dens = self.log_densities(log_returns)
        shift = np.max(log_dens, axis=-1, keepdims=True)  # Avoid underflow when every density is tiny
        unnormalized = predicted * np.exp(log_dens - shift)
        normalizer = np.sum(unnormalized, axis=-1, keepdims=True)
        predictive_log_density = np.log(np.maximum(normalizer[..., 0], TINY)) + shift[..., 0]

        return unnormalized / np.maximum(normalizer, TINY), predictive_log_density

    def simulate(self, n_periods, rng, initial_regime=None):
        # Returns (regimes, log_returns) with shapes (n_periods,) and (n_periods, N)
        regimes = np.zeros(n_periods, dtype=int)
        regimes[0] = rng.choice(self.n_regimes, p=self.initial_distribution) if initial_regime is None else initial_regime
        for k in range(1, n_periods):
            regimes[k] = rng.choice(self.n_regimes, p=self.transition_matrix[regimes[k - 1]])

        return regimes, self.simulate_log_returns(regimes, rng)

    def simulate_log_returns(self, regimes, rng):
        # Draw log returns conditional on a vector of regimes (any shape); returns shape regimes.shape + (N,)
        # The "Cholesky" factor is computed by hand from the eigendecomposition so that riskless assets
        # (zero variance) are allowed
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariances)
        root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))[:, np.newaxis, :]  # root @ root.T = covariance
        eta = rng.standard_normal(np.shape(regimes) + (self.n_assets,))

        return self.means[regimes] + np.einsum("...ij,...j->...i", root[regimes], eta)


def stationary_distribution(transition_matrix):

    # The probability vector pi with pi @ P = pi (the left eigenvector of P with eigenvalue 1)
    P = np.atleast_2d(transition_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    stationary = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1))])
    stationary = np.clip(stationary / stationary.sum(), 0, None)

    return stationary / stationary.sum()


def forward_backward(hmm, log_returns):

    # Scaled forward-backward algorithm (van Handel algorithms 3.2 and 3.4) for a single series of log returns
    # with shape (T, N). Returns a dict with:
    #   filtered: (T, K), Pr(X_k | Y_1, ..., Y_k)
    #   smoothed: (T, K), Pr(X_k | Y_1, ..., Y_T)
    #   pairwise: (T - 1, K, K), Pr(X_k = i, X_{k+1} = j | Y_1, ..., Y_T)
    #   predictive_log_density: (T,), log Pr(Y_k | Y_1, ..., Y_{k-1}), whose sum is the log likelihood
    y = np.atleast_2d(np.asarray(log_returns, dtype=float))
    T, K = y.shape[0], hmm.n_regimes
    P = hmm.transition_matrix

    log_dens = hmm.log_densities(y)  # (T, K)
    shift = np.max(log_dens, axis=1, keepdims=True)
    upsilon = np.exp(log_dens - shift)  # Rescaled densities: the rescaling cancels in the normalized quantities

    filtered = np.zeros((T, K))
    scale = np.zeros(T)
    unnormalized = hmm.initial_distribution * upsilon[0]
    for k in range(T):
        if k > 0:
            unnormalized = (filtered[k - 1] @ P) * upsilon[k]
        scale[k] = unnormalized.sum()
        filtered[k] = unnormalized / scale[k]

    beta = np.ones((T, K))
    pairwise = np.zeros((T - 1, K, K))
    for k in reversed(range(T - 1)):
        beta[k] = P @ (upsilon[k + 1] * beta[k + 1]) / scale[k + 1]
        pairwise[k] = filtered[k][:, np.newaxis] * P * (upsilon[k + 1] * beta[k + 1])[np.newaxis, :] / scale[k + 1]

    smoothed = filtered * beta

    return {
        "filtered": filtered,
        "smoothed": smoothed,
        "pairwise": pairwise,
        "predictive_log_density": np.log(scale) + shift[:, 0],
    }


def em_step(hmm, log_returns):

    # One iteration of Baum-Welch (EM, van Handel algorithm 6.1) for a single series of log returns
    # Returns the updated HMM and the log likelihood of the data under the _input_ HMM
    y = np.atleast_2d(np.asarray(log_returns, dtype=float))
    fb = forward_backward(hmm, y)
    smoothed, pairwise = fb["smoothed"], fb["pairwise"]

    transition_matrix = pairwise.sum(axis=0) / smoothed[:-1].sum(axis=0)[:, np.newaxis]
    weights = smoothed.sum(axis=0)  # Expected number of periods in each regime
    means = (smoothed.T @ y) / weights[:, np.newaxis]
    deviations = y[np.newaxis, :, :] - means[:, np.newaxis, :]  # (K, T, N)
    covariances = np.einsum("tk,kti,ktj->kij", smoothed, deviations, deviations) / weights[:, np.newaxis, np.newaxis]

    updated = GaussianHMM(
        transition_matrix=transition_matrix / transition_matrix.sum(axis=1, keepdims=True),
        means=means,
        covariances=covariances,
        initial_distribution=smoothed[0],
        regime_names=hmm.regime_names,
    )

    return updated, fb["predictive_log_density"].sum()


def random_initial_hmm(n_regimes, log_returns, rng, diag_min=0.9, diag_max=0.999):

    # Same recipe as get_random_initial_parameters_gaussian in the R code: P close to the identity (regimes are
    # persistent), means spread around the empirical quantiles, covariances around the empirical covariance
    y = np.atleast_2d(np.asarray(log_returns, dtype=float))
    K, N = n_regimes, y.shape[1]

    if K == 1:
        transition_matrix = np.ones((1, 1))
    else:
        diagonal = rng.uniform(diag_min, diag_max, size=K)
        transition_matrix = np.tile(((1 - diagonal) / (K - 1))[:, np.newaxis], (1, K))
        np.fill_diagonal(transition_matrix, diagonal)

    quantiles = np.quantile(y, np.linspace(0, 1, K + 2)[1:-1], axis=0)  # (K, N)
    means = quantiles + rng.normal(scale=0.1 * y.std(axis=0), size=(K, N))
    covariances = np.cov(y, rowvar=False).reshape(N, N)[np.newaxis] * rng.uniform(0.5, 1.5, size=(K, 1, 1))

    initial_distribution = rng.uniform(size=K)

    return GaussianHMM(transition_matrix, means, covariances, initial_distribution / initial_distribution.sum())


def fit_gaussian_hmm(log_returns, n_regimes, rng, n_random_starts=5, max_iter=200, tol=1e-6):

    # Fit a Gaussian HMM by EM from several random starting points, and return the fit with the highest log likelihood
    # (the R code's fit_gaussian_hmm). Regimes are relabeled in order of increasing volatility of the last asset
    # (EM only identifies regimes up to a relabeling)
    best = None
    for _ in range(n_random_starts):
        hmm = random_initial_hmm(n_regimes, log_returns, rng)
        loglik_path = []
        for _ in range(max_iter):
            hmm, loglik = em_step(hmm, log_returns)
            loglik_path.append(loglik)
            if len(loglik_path) > 1 and abs(loglik_path[-1] - loglik_path[-2]) < tol * abs(loglik_path[-1]):
                break
        hmm.loglik_path = loglik_path
        if best is None or loglik_path[-1] > best.loglik_path[-1]:
            best = hmm

    return relabel_regimes_by_volatility(best, rename=True)


def relabel_regimes_by_volatility(hmm, asset_idx=-1, rename=False):

    # With rename=True, the relabeled regimes are named "regime 0" (least volatile), "regime 1", ...
    order = np.argsort(hmm.covariances[:, asset_idx, asset_idx])
    relabeled = GaussianHMM(
        transition_matrix=hmm.transition_matrix[np.ix_(order, order)],
        means=hmm.means[order],
        covariances=hmm.covariances[order],
        initial_distribution=hmm.initial_distribution[order],
        regime_names=None if rename else tuple(hmm.regime_names[i] for i in order),
    )
    relabeled.loglik_path = hmm.loglik_path

    return relabeled
