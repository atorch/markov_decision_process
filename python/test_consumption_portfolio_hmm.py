import numpy as np
import pytest

import consumption_savings
from consumption_portfolio_hmm import (
    LifecycleModel,
    Market,
    bank_and_equity_market,
    belief_vector,
    interpolate,
    iid_version,
)
from hmm import GaussianHMM


def test_interpolate_is_exact_for_multilinear_functions():

    axes = [np.geomspace(0.1, 10, 7), np.linspace(0, 1, 5)]
    x, y = np.meshgrid(*axes, indexing="ij")
    table = 2 + 3 * x - y + 0.5 * x * y
    points_x, points_y = np.array([0.05, 0.3, 4.4, 12.0]), np.array([0.0, 0.33, 0.9, 1.0])

    np.testing.assert_allclose(
        interpolate(axes, table, [points_x, points_y]), 2 + 3 * points_x - points_y + 0.5 * points_x * points_y
    )


@pytest.fixture(scope="module")
def benchmark_policy():

    return consumption_savings.get_greedy_policy(consumption_savings.solve_by_backward_induction())


def test_bank_only_log_utility_matches_consumption_savings_benchmark(benchmark_policy):

    # With log utility and no equities, this is exactly the model in consumption_savings.py,
    # solved on a different (normalized) state space. The two solutions should agree
    bank = np.log(1 + consumption_savings.INTEREST_RATE)
    market = Market("bank", ("bank", "equities"), GaussianHMM([[1.0]], [[bank, 0.05]], [np.diag([0.0, 0.2**2])]))
    model = LifecycleModel(market, risk_aversion=1.0, max_risky_share=0.0).solve(verbose=False)

    wealth = np.array([0.0, 0.5, 2.0, 8.0])
    for age in [25, 40, 55, 64, 65, 80]:
        income = np.full(wealth.size, 0.0 if age >= consumption_savings.RETIREMENT_AGE else 1.3)
        consumption, risky_share = model.policy(wealth + (income == 0) * 0.1, income, np.ones((wealth.size, 1)), age)
        expected = benchmark_policy(wealth + (income == 0) * 0.1, income, age) * (wealth + (income == 0) * 0.1 + income)
        np.testing.assert_allclose(consumption, expected, rtol=0.01, err_msg=f"age {age}")
        np.testing.assert_array_equal(risky_share, 0.0)


def myopic_risky_share(model, beliefs, objective):

    # Brute force: the share that maximizes E[objective(portfolio return)] over next year's regime and returns
    shares = np.linspace(0, 1, 2001)
    predicted = model.hmm.predict(beliefs)  # (n, K)
    gross = model.gross_return_nodes  # (K, Q, 2)
    portfolio = (1 - shares[:, None, None]) * gross[None, :, :, 0] + shares[:, None, None] * gross[None, :, :, 1]
    expected = np.einsum("nk,skq,q->ns", predicted, objective(portfolio), model.quadrature_weights)

    return shares[np.argmax(expected, axis=1)]


def test_log_utility_retirees_are_myopic():

    # Samuelson (1969) and Merton (1969): with log utility and no labor income, the optimal portfolio maximizes the
    # expected log return over the next year, even when returns are predictable (here, through the belief),
    # and the consumption rate is 1 / (remaining sum of discount factors)
    model = LifecycleModel(bank_and_equity_market(), risk_aversion=1.0).solve(verbose=False)
    beliefs = belief_vector([0.0, 0.3, 0.6, 1.0])
    expected = myopic_risky_share(model, beliefs, np.log)

    for age in [65, 75, 84]:
        fraction, risky_share, _ = model.retired_policy(beliefs, age)
        np.testing.assert_allclose(risky_share, expected, atol=0.004, err_msg=f"age {age}")
        np.testing.assert_allclose(fraction, 1 / consumption_savings.remaining_lifetime_discount_sum(age))


def test_one_year_before_death_retirees_are_myopic():

    # At MAX_AGE - 1 there's no future belief to hedge against (the agent consumes everything next year, whatever
    # the regime), so for any risk aversion the portfolio maximizes E[portfolio return ^ (1 - gamma)] / (1 - gamma)
    gamma = 5.0
    model = LifecycleModel(bank_and_equity_market(), risk_aversion=gamma)
    model.solve(verbose=False)
    beliefs = belief_vector([0.0, 0.3, 0.6, 1.0])
    expected = myopic_risky_share(model, beliefs, lambda r: r ** (1 - gamma) / (1 - gamma))

    _, risky_share, _ = model.retired_policy(beliefs, consumption_savings.MAX_AGE - 1)
    np.testing.assert_allclose(risky_share, expected, atol=0.004)


def test_iid_version_matches_long_run_moments():

    market = bank_and_equity_market()
    iid = iid_version(market)
    weights = market.hmm.initial_distribution
    rng = np.random.default_rng(0)
    regimes = rng.choice(2, p=weights, size=400_000)
    draws = market.hmm.simulate_log_returns(regimes, rng)

    np.testing.assert_allclose(iid.hmm.means[0], draws.mean(axis=0), atol=0.002)
    np.testing.assert_allclose(iid.hmm.covariances[0], np.cov(draws, rowvar=False), atol=0.002)
