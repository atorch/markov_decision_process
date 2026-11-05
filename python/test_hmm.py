import itertools

import numpy as np
import pytest

from hmm import GaussianHMM, fit_gaussian_hmm, forward_backward, stationary_distribution


def two_asset_hmm():

    # Bonds and equities, calm and turbulent regimes (the same shape as the lifecycle model's calibration)
    return GaussianHMM(
        transition_matrix=[[0.95, 0.05], [0.15, 0.85]],
        means=[[0.02, 0.08], [0.03, -0.02]],
        covariances=[
            [[0.05**2, 0.2 * 0.05 * 0.12], [0.2 * 0.05 * 0.12, 0.12**2]],
            [[0.07**2, -0.3 * 0.07 * 0.30], [-0.3 * 0.07 * 0.30, 0.30**2]],
        ],
    )


def test_stationary_distribution():

    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    pi = stationary_distribution(P)
    np.testing.assert_allclose(pi @ P, pi)
    np.testing.assert_allclose(pi, [0.75, 0.25])


def test_forward_backward_matches_brute_force_enumeration():

    # On a short series, enumerate every regime path to get exact filtered and smoothed probabilities
    hmm = two_asset_hmm()
    _, y = hmm.simulate(5, np.random.default_rng(1))
    densities = np.exp(hmm.log_densities(y))

    joint = {}
    for path in itertools.product(range(hmm.n_regimes), repeat=len(y)):
        probability = hmm.initial_distribution[path[0]] * densities[0, path[0]]
        for k in range(1, len(y)):
            probability *= hmm.transition_matrix[path[k - 1], path[k]] * densities[k, path[k]]
        joint[path] = probability

    likelihood = sum(joint.values())
    smoothed = np.array([[sum(p for path, p in joint.items() if path[k] == i) for i in range(2)] for k in range(len(y))])
    smoothed /= likelihood

    fb = forward_backward(hmm, y)
    np.testing.assert_allclose(fb["smoothed"], smoothed, rtol=1e-10)
    np.testing.assert_allclose(fb["predictive_log_density"].sum(), np.log(likelihood), rtol=1e-10)

    # The last filtered belief is the last smoothed belief
    np.testing.assert_allclose(fb["filtered"][-1], smoothed[-1], rtol=1e-10)


def test_sequential_update_matches_forward_pass():

    # The agent's one-step-at-a-time belief update is the forward pass of the forward-backward algorithm
    hmm = two_asset_hmm()
    _, y = hmm.simulate(50, np.random.default_rng(2))
    fb = forward_backward(hmm, y)

    belief = fb["filtered"][0]
    for k in range(1, len(y)):
        belief, log_density = hmm.update(belief, y[k])
        np.testing.assert_allclose(belief, fb["filtered"][k], rtol=1e-10)
        np.testing.assert_allclose(log_density, fb["predictive_log_density"][k], rtol=1e-10)


def test_riskless_asset_is_ignored_by_the_filter():

    # A bank account with a fixed interest rate says nothing about the regime: the filter should give the same
    # beliefs as a model of equities alone
    equity_only = GaussianHMM([[0.9, 0.1], [0.2, 0.8]], means=[[0.08], [-0.02]], covariances=[[[0.12**2]], [[0.3**2]]])
    with_bank = GaussianHMM(
        [[0.9, 0.1], [0.2, 0.8]],
        means=[[0.01, 0.08], [0.01, -0.02]],
        covariances=[np.diag([0.0, 0.12**2]), np.diag([0.0, 0.3**2])],
    )
    _, y = with_bank.simulate(30, np.random.default_rng(3))
    assert np.all(y[:, 0] == 0.01)
    np.testing.assert_allclose(forward_backward(with_bank, y)["filtered"], forward_backward(equity_only, y[:, 1:])["filtered"])


def test_one_regime_hmm_is_iid_gaussian_mle():

    # Same sanity check as the R code: with a single regime there are no regime dynamics,
    # so EM recovers the i.i.d. Gaussian MLE (sample mean and covariance with an n, not n - 1, denominator)
    y = np.random.default_rng(4).multivariate_normal([0.02, 0.07], [[0.003, 0.001], [0.001, 0.03]], size=500)
    fit = fit_gaussian_hmm(y, n_regimes=1, rng=np.random.default_rng(5), n_random_starts=1)
    np.testing.assert_allclose(fit.means[0], y.mean(axis=0))
    np.testing.assert_allclose(fit.covariances[0], np.cov(y, rowvar=False, bias=True))


@pytest.mark.parametrize("seed", [6])
def test_em_recovers_simulated_parameters(seed):

    hmm = two_asset_hmm()
    _, y = hmm.simulate(20_000, np.random.default_rng(seed))
    fit = fit_gaussian_hmm(y, n_regimes=2, rng=np.random.default_rng(seed + 1), n_random_starts=3)

    # The log likelihood never decreases during EM
    assert np.all(np.diff(fit.loglik_path) > -1e-6)

    np.testing.assert_allclose(fit.transition_matrix, hmm.transition_matrix, atol=0.02)
    np.testing.assert_allclose(fit.means, hmm.means, atol=0.01)
    np.testing.assert_allclose(np.sqrt(np.diagonal(fit.covariances, axis1=1, axis2=2)),
                               np.sqrt(np.diagonal(hmm.covariances, axis1=1, axis2=2)), atol=0.01)
