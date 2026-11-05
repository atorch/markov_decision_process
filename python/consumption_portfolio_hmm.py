"""Lifecycle consumption, savings and portfolio choice when asset returns follow a hidden Markov model.

This extends consumption_savings.py: same lifespan, retirement age, income process, discount factor and borrowing
constraint, but every year the agent also chooses how to split their savings between a safer asset (a bank account,
or bonds) and equities. Returns depend on a hidden regime (calm or turbulent) that follows a Markov chain. The agent
never sees the regime: they know the HMM's parameters, observe realized returns, and update their belief about the
regime with Bayes' rule (the forward algorithm in hmm.py).

That makes the problem a partially observed MDP. The standard trick is to make the belief part of the state:
the state is (wealth, income, age, belief), the belief is a probability vector over the K regimes, and the belief
evolves according to the filter, which is a known (if nonlinear) function of the old belief and the new returns.

Two more simplifications keep the state space small enough for backward induction on a grid:

* Homotheticity. With CRRA utility and income that only ever gets multiplied by (1 + raise), scaling wealth and income
  by the same factor scales every future consumption level by that factor, so V(wealth, income, age, belief) =
  income^(1 - gamma) * v(wealth / income, age, belief). The working-age state is (wealth / income, belief).
  Retirees have no income, so V = wealth^(1 - gamma) * u(age, belief): their consumption rate and portfolio
  don't depend on how rich they are, only on their age and belief.
* Post-decision states (Sutton and Barto's "afterstates", sec. 6.8). The portfolio choice only depends on how much the
  agent saves (as a multiple of income), not on how they got there, so we first solve for the best portfolio and the
  expected continuation value on a grid of (savings / income, belief), and then choose consumption by searching
  over a one dimensional grid. This is much cheaper than searching over (consumption, portfolio) pairs.

Optionally (see EmploymentRisk), the worker can lose their job. Employment follows a Markov chain whose transition
probabilities can depend on the hidden regime, so job losses can be more likely in turbulent years. The agent observes
their own employment status, so it's part of the state, and when the transition probabilities depend on the regime,
losing (or finding) a job is also news about the regime, and the agent's belief takes it into account. To keep
homotheticity, income is tracked as a wage that the worker earns while employed: the unemployed receive a fraction of
it (unemployment benefits), it doesn't grow while they're unemployed, and they get it back (minus a "scar") when they
find a new job. The working-age state is then (wealth / wage, employment status, belief).
"""

import argparse
import itertools
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consumption_savings import (
    AGE_GRID,
    DISCOUNT_FACTOR,
    INITIAL_INCOME,
    INITIAL_WEALTH,
    INTEREST_RATE,
    MAX_AGE,
    MAX_RAISE,
    MIN_AGE,
    MIN_RAISE,
    PLOT_STYLE,
    RAISE_PROBABILITY,
    RAISE_RATE_PROBABILITIES,
    RAISE_RATES,
    RETIREMENT_AGE,
    get_plot_path,
    plot_median_and_band,
    remaining_lifetime_discount_sum,
    shade_retirement,
)
from hmm import GaussianHMM, stationary_distribution

# Coefficient of relative risk aversion. consumption_savings.py uses log utility (gamma = 1), but a log utility agent
# is a very aggressive investor: with these return parameters they'd hold 100% equities in calm times no matter what,
# and their portfolio choice is myopic (it ignores what this year's returns say about future returns). Gamma > 1
# gives interior portfolios and a hedging motive
RISK_AVERSION = 5.0

# Grids for the working-age value fn (wealth / income) and for post-decision savings (savings / income)
# Denser at low wealth, where the value fn has the most curvature
WEALTH_RATIO_GRID = np.concatenate([[0.0], np.geomspace(0.01, 60.0, 60)])
SAVINGS_RATIO_GRID = np.geomspace(1e-3, 60.0, 60)

# Grid points per belief coordinate (the belief over K regimes has K - 1 free coordinates)
BELIEF_POINTS_PER_AXIS = 21

# Fraction of savings in the risky asset (equities): no shorting, no leverage
RISKY_SHARE_GRID = np.linspace(0.0, 1.0, 11)
CONSUMPTION_FRACTION_GRID = np.linspace(0.01, 1.0, 100)
MIN_CONSUMPTION_FRACTION = 0.001

# Gauss-Hermite nodes per dimension for the expectation over (normally distributed) log returns
QUADRATURE_NODES = {1: 9, 2: 6}

TINY = 1e-12

# Plot colors: categorical slots of a colorblind-safe palette, plus a sequential blue ramp for beliefs
CALM_COLOR = "#2a78d6"
TURBULENT_COLOR = "#eb6834"
IID_COLOR = "#52514e"
INCOME_COLOR = "#1baf7a"
BELIEF_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


@dataclass
class Market:

    # Two assets: index 0 is the safer asset (a bank account or bonds), index 1 is equities
    name: str
    asset_names: tuple[str, str]
    hmm: GaussianHMM


def _covariance(vol_safe, vol_equity, correlation):

    return np.array(
        [[vol_safe**2, correlation * vol_safe * vol_equity], [correlation * vol_safe * vol_equity, vol_equity**2]]
    )


# Annual log returns. Calm years: equities return a lot on average, with modest volatility.
# Turbulent years: equities are about twice as volatile, with a much lower mean.
# Regimes are persistent: a calm year is followed by another calm year with probability 0.9,
# and a turbulent year is followed by another turbulent year with probability 0.75 (so turbulent years are
# about 29% of all years in the long run, and a turbulent spell lasts 4 years on average)
TRANSITION_MATRIX = np.array([[0.9, 0.1], [0.25, 0.75]])
REGIME_NAMES = ("calm", "turbulent")
EQUITY_MEANS, EQUITY_VOLS = (0.09, -0.02), (0.13, 0.28)


def bank_and_equity_market():

    # The bank pays INTEREST_RATE (the same as in consumption_savings.py) in every regime, without risk
    bank = np.log(1 + INTEREST_RATE)
    hmm = GaussianHMM(
        transition_matrix=TRANSITION_MATRIX,
        means=[[bank, EQUITY_MEANS[0]], [bank, EQUITY_MEANS[1]]],
        covariances=[_covariance(0.0, EQUITY_VOLS[0], 0.0), _covariance(0.0, EQUITY_VOLS[1], 0.0)],
        regime_names=REGIME_NAMES,
    )

    return Market("bank and equities", ("bank", "equities"), hmm)


def bond_and_equity_market():

    # Bonds are risky, but less risky than equities. In calm years bond and equity returns are mildly positively
    # correlated; in turbulent years they're negatively correlated (a flight to quality), and bonds return a bit more
    hmm = GaussianHMM(
        transition_matrix=TRANSITION_MATRIX,
        means=[[0.02, EQUITY_MEANS[0]], [0.03, EQUITY_MEANS[1]]],
        covariances=[_covariance(0.06, EQUITY_VOLS[0], 0.2), _covariance(0.08, EQUITY_VOLS[1], -0.4)],
        regime_names=REGIME_NAMES,
    )

    return Market("bonds and equities", ("bonds", "equities"), hmm)


def iid_version(market):

    # A single regime with the same (long run) mean and covariance of log returns as the regime-switching market,
    # for comparison: what changes when the agent believes returns are i.i.d.?
    hmm = market.hmm
    weights = hmm.initial_distribution  # The stationary distribution
    mean = weights @ hmm.means
    deviations = hmm.means - mean
    covariance = np.einsum("k,kij->ij", weights, hmm.covariances) + np.einsum("k,ki,kj->ij", weights, deviations, deviations)

    return Market(f"{market.name} (i.i.d.)", market.asset_names, GaussianHMM([[1.0]], [mean], [covariance]))


# Employment risk. Index 0 is employed, index 1 is unemployed. Every year, an employed worker loses their job with
# a separation probability, and an unemployed worker finds a job with a job finding probability. Both can depend on
# the regime of the year in which the transition happens, which is the same regime that generates that year's returns:
# in turbulent years, job losses are more likely and jobs are harder to find. If a regime lasted forever, its
# unemployment rate would converge to about 5% (calm) or 10% (turbulent). Because regimes switch, the average
# unemployment rates in calm and turbulent years are about 5% and 9%, and 6% overall
EMPLOYED, UNEMPLOYED = 0, 1
SEPARATION_PROBABILITIES = (0.03, 0.05)
JOB_FINDING_PROBABILITIES = (0.6, 0.45)

# Unemployment benefits (and any other income while unemployed), as a fraction of the wage
REPLACEMENT_RATE = 0.4

# A worker who finds a new job is paid their old wage (which doesn't grow while they're unemployed) times this factor.
# Displaced workers' earnings stay well below what they would have been for many years (Jacobson, LaLonde and Sullivan
# 1993; Davis and von Wachter 2011), and in this model the scar is permanent
REEMPLOYMENT_WAGE_FACTOR = 0.9


@dataclass
class EmploymentRisk:

    separation_probabilities: tuple  # (K,): Pr(employed -> unemployed | regime)
    job_finding_probabilities: tuple  # (K,): Pr(unemployed -> employed | regime)
    replacement_rate: float = REPLACEMENT_RATE
    reemployment_wage_factor: float = REEMPLOYMENT_WAGE_FACTOR

    # Does the agent use their own employment transitions to update their belief about the regime?
    # (This only matters when the transition probabilities depend on the regime)
    learn_from_employment: bool = True

    def __post_init__(self):
        self.separation_probabilities = np.asarray(self.separation_probabilities, dtype=float)
        self.job_finding_probabilities = np.asarray(self.job_finding_probabilities, dtype=float)
        assert self.separation_probabilities.shape == self.job_finding_probabilities.shape
        assert np.all((0 < self.separation_probabilities) & (self.separation_probabilities < 1))
        assert np.all((0 < self.job_finding_probabilities) & (self.job_finding_probabilities < 1))

    @property
    def n_regimes(self):
        return self.separation_probabilities.size

    def transition_matrices(self):

        # (K, 2, 2): Pr(next employment status | current status, next year's regime)
        separation, finding = self.separation_probabilities, self.job_finding_probabilities
        return np.stack([np.stack([1 - separation, separation], axis=-1), np.stack([finding, 1 - finding], axis=-1)], axis=1)

    def joint_stationary_distribution(self, hmm):

        # (K, 2): the long run distribution of (regime, employment status). The pair is a Markov chain: next year's
        # regime depends on this year's, and next year's employment status depends on this year's and on next year's regime
        employment = self.transition_matrices().transpose(1, 0, 2)  # (j, k', j')
        transitions = hmm.transition_matrix[:, np.newaxis, :, np.newaxis] * employment[np.newaxis]  # (k, j, k', j')
        n_regimes = hmm.n_regimes
        joint = transitions.reshape(n_regimes * 2, n_regimes * 2)  # Rows: (k, j), columns: (k', j')

        return stationary_distribution(joint).reshape(n_regimes, 2)

    def long_run_unemployment_rate(self, hmm):
        return self.joint_stationary_distribution(hmm)[:, UNEMPLOYED].sum()

    def regime_independent(self, hmm, n_regimes=None):

        # The same transition probabilities in every regime, so that employment risk is independent of returns and
        # carries no information about the regime. n_regimes=1 gives a version for an i.i.d. market.
        # Each probability is the long run average of the regime-dependent one over the years in which it applies:
        # the separation probability is averaged over next year's regime for workers who are employed this year,
        # and the job finding probability over next year's regime for workers who are unemployed this year. In the
        # long run, flows into and out of unemployment balance, so this matches both the long run unemployment rate
        # and the frequency of job losses: only the correlation between unemployment and the regime changes
        joint = self.joint_stationary_distribution(hmm)  # (K, 2)
        next_regime = joint.T @ hmm.transition_matrix  # (2, K): Pr(this year's status, next year's regime)
        next_regime = next_regime / next_regime.sum(axis=1, keepdims=True)
        n_regimes = hmm.n_regimes if n_regimes is None else n_regimes

        return EmploymentRisk(
            np.full(n_regimes, next_regime[EMPLOYED] @ self.separation_probabilities),
            np.full(n_regimes, next_regime[UNEMPLOYED] @ self.job_finding_probabilities),
            self.replacement_rate,
            self.reemployment_wage_factor,
            self.learn_from_employment,
        )

    def long_run_unemployment_rates(self):

        # The unemployment rate that each regime would converge to if it lasted forever
        return self.separation_probabilities / (self.separation_probabilities + self.job_finding_probabilities)


def regime_dependent_employment_risk(**kwargs):
    return EmploymentRisk(SEPARATION_PROBABILITIES, JOB_FINDING_PROBABILITIES, **kwargs)


class BeliefGrid:

    # A grid over beliefs (probability vectors over K regimes), parameterized by the probabilities of regimes 1, ..., K - 1
    # (the probability of regime 0 is whatever is left). For K = 2 this is a grid over Pr(turbulent).
    # For K > 2 the product grid includes points off the simplex; their beliefs are projected back onto it
    def __init__(self, n_regimes, points_per_axis=BELIEF_POINTS_PER_AXIS):
        self.axes = [np.linspace(0.0, 1.0, points_per_axis)] * (n_regimes - 1)
        self.shape = tuple(len(axis) for axis in self.axes)

        if n_regimes == 1:
            coordinates = np.zeros((1, 0))  # A single belief: the one regime has probability 1
        else:
            coordinates = np.stack([m.ravel() for m in np.meshgrid(*self.axes, indexing="ij")], axis=-1)
        beliefs = np.concatenate([1 - coordinates.sum(axis=1, keepdims=True), coordinates], axis=1)
        beliefs = np.clip(beliefs, 0.0, None)
        self.beliefs = beliefs / beliefs.sum(axis=1, keepdims=True)  # (number of grid points, K)


def interpolate(axes, table, coordinates):

    # Multilinear interpolation (linear extrapolation outside of the grid) of a table of values on a product grid
    # axes: list of d increasing 1d arrays, table: array of shape (len(axes[0]), ..., len(axes[d - 1]))
    # coordinates: list of d arrays that broadcast against each other. Returns an array with their broadcast shape
    if not axes:
        return table[()]

    indices, weights = [], []
    for axis, coordinate in zip(axes, coordinates):
        idx = np.clip(np.searchsorted(axis, coordinate, side="right") - 1, 0, len(axis) - 2)
        indices.append(idx)
        weights.append((coordinate - axis[idx]) / (axis[idx + 1] - axis[idx]))

    result = 0.0
    for corner in itertools.product([0, 1], repeat=len(axes)):
        weight = 1.0
        for w, bit in zip(weights, corner):
            weight = weight * (w if bit else 1 - w)
        result = result + weight * table[tuple(idx + bit for idx, bit in zip(indices, corner))]

    return result


class LifecycleModel:

    def __init__(self, market, risk_aversion=RISK_AVERSION, max_risky_share=1.0, employment=None):
        self.market = market
        self.hmm = market.hmm
        self.gamma = risk_aversion
        self.risky_share_grid = RISKY_SHARE_GRID * max_risky_share
        self.belief_grid = BeliefGrid(self.hmm.n_regimes)
        self._build_quadrature()
        self._build_employment(employment)

        # Solution, filled in by solve(). Working ages: inverted value fn on (employment, wealth ratio, belief) and
        # inverted post-decision value on (employment, savings ratio, belief). Retirement ages: value coefficient u(age, belief)
        self.inverted_value = {}
        self.inverted_post_decision_value = {}
        self.retired_value = {}

    def _build_employment(self, employment):

        # Employment statuses j = 0, ..., J - 1, with J = 1 (always employed) when there's no employment risk
        # employment_transitions[k, j, j'] = Pr(next status j' | current status j, next year's regime k)
        # income_multipliers[j]: income as a multiple of the wage. wage_growth[j][j']: (growth factors, probabilities)
        # for the wage, from one year to the next: employed workers get random raises, the unemployed don't, and a
        # worker who finds a job gets their old wage times the reemployment factor
        self.employment = employment
        raises = (1 + RAISE_RATES, RAISE_RATE_PROBABILITIES)
        if employment is None:
            self.employment_transitions = np.ones((self.hmm.n_regimes, 1, 1))
            self.income_multipliers = np.ones(1)
            self.wage_growth = [[raises]]
            self.learn_from_employment = False
        else:
            assert employment.n_regimes == self.hmm.n_regimes
            self.employment_transitions = employment.transition_matrices()
            self.income_multipliers = np.array([1.0, employment.replacement_rate])
            no_change, reemployed = (np.ones(1), np.ones(1)), (np.array([employment.reemployment_wage_factor]), np.ones(1))
            self.wage_growth = [[raises, no_change], [reemployed, no_change]]
            self.learn_from_employment = employment.learn_from_employment
        self.n_employment_states = self.income_multipliers.size

    def _build_quadrature(self):

        # Discretize each regime's distribution of log returns with a product of Gauss-Hermite nodes over the
        # informative (risky) assets. Riskless assets sit at their mean
        hmm = self.hmm
        risky = hmm.informative_assets()
        n_nodes = QUADRATURE_NODES[risky.size]
        nodes_1d, weights_1d = np.polynomial.hermite_e.hermegauss(n_nodes)
        weights_1d = weights_1d / weights_1d.sum()
        z = np.array(list(itertools.product(nodes_1d, repeat=risky.size)))  # (Q, number of risky assets)
        self.quadrature_weights = np.prod(np.array(list(itertools.product(weights_1d, repeat=risky.size))), axis=1)

        log_returns = np.tile(hmm.means[:, np.newaxis, :], (1, z.shape[0], 1))  # (K, Q, N)
        for k in range(hmm.n_regimes):
            cholesky = np.linalg.cholesky(hmm.covariances[k][np.ix_(risky, risky)])
            log_returns[k][:, risky] += z @ cholesky.T

        self.log_return_nodes = log_returns
        self.gross_return_nodes = np.exp(log_returns)

    def update_beliefs(self, beliefs, log_returns, employment=None, next_employment=None):

        # The filter: next year's belief after observing next year's returns and, when the agent learns from it,
        # their own employment transition. Vectorized: beliefs (..., K), log_returns (..., N), employment and
        # next_employment are ints or arrays that broadcast against beliefs[..., 0]
        next_beliefs, _ = self.hmm.update(beliefs, log_returns)
        if self.learn_from_employment and employment is not None:
            # Bayes' rule again: multiply by the likelihood of the employment transition under each regime
            likelihood = np.moveaxis(self.employment_transitions[:, employment, next_employment], 0, -1)
            next_beliefs = next_beliefs * likelihood
            next_beliefs = next_beliefs / next_beliefs.sum(axis=-1, keepdims=True)

        return next_beliefs

    # Utility and value fn transformations

    def utility(self, consumption):

        consumption = np.maximum(consumption, TINY)
        if self.gamma == 1:
            return np.log(consumption)

        return consumption ** (1 - self.gamma) / (1 - self.gamma)

    def invert(self, value, scale):

        # The "inverted" value fn is close to linear in wealth, so it interpolates much better than the value fn itself
        # (see the benchmark in consumption_savings.py). For CRRA utility it's the certainty equivalent ((1 - gamma) V) ^ (1 / (1 - gamma)),
        # which is exactly linear in total wealth without uncertainty; for log utility it's exp(V / scale), where
        # scale is the remaining sum of discount factors
        if self.gamma == 1:
            return np.exp(value / scale)

        return ((1 - self.gamma) * value) ** (1 / (1 - self.gamma))

    def uninvert(self, inverted, scale):

        inverted = np.maximum(inverted, TINY)
        if self.gamma == 1:
            return scale * np.log(inverted)

        return inverted ** (1 - self.gamma) / (1 - self.gamma)

    # Bellman equation pieces

    def value_scale(self, age):
        return remaining_lifetime_discount_sum(age)

    def post_decision_scale(self, age):
        return DISCOUNT_FACTOR * remaining_lifetime_discount_sum(age + 1)

    def working_value(self, wealth_ratio, beliefs, age, employment=EMPLOYED):

        # v_j(wealth / wage, age, belief) by interpolation; beliefs has shape (..., K), employment j is an int
        coordinates = [wealth_ratio] + [beliefs[..., j] for j in range(1, self.hmm.n_regimes)]
        inverted = interpolate([WEALTH_RATIO_GRID] + self.belief_grid.axes, self.inverted_value[age][employment], coordinates)

        return self.uninvert(inverted, self.value_scale(age))

    def post_decision_value(self, savings_ratio, beliefs, age, employment=EMPLOYED):

        coordinates = [savings_ratio] + [beliefs[..., j] for j in range(1, self.hmm.n_regimes)]
        inverted = interpolate(
            [SAVINGS_RATIO_GRID] + self.belief_grid.axes, self.inverted_post_decision_value[age][employment], coordinates
        )

        return self.uninvert(inverted, self.post_decision_scale(age))

    def retired_value_coefficient(self, beliefs, age):

        coordinates = [beliefs[..., j] for j in range(1, self.hmm.n_regimes)]
        return interpolate(self.belief_grid.axes, self.retired_value[age], coordinates)

    def expected_next_value(self, savings_ratio, beliefs, risky_shares, age, employment=EMPLOYED):

        # beta * E[V(next state)] after saving savings_ratio (a multiple of the wage, or of wealth for retirees, in
        # which case savings_ratio is 1) and putting the fraction risky_shares of savings in equities
        # Shapes: savings_ratio (n,), beliefs (n, K), risky_shares (n, m). Returns (n, m)
        # The expectation is over next year's regime (given the current belief), returns given that regime
        # (quadrature nodes), next year's employment status given that regime, and raises.
        # Next year's belief depends on the returns (and the employment transition), but not on the agent's actions
        hmm, gamma, next_age = self.hmm, self.gamma, age + 1

        predicted = hmm.predict(beliefs)  # (n, K): Pr(next year's regime)
        safe_return = self.gross_return_nodes[np.newaxis, np.newaxis, :, :, 0]
        risky_return = self.gross_return_nodes[np.newaxis, np.newaxis, :, :, 1]
        alpha = risky_shares[:, :, np.newaxis, np.newaxis]
        portfolio_return = (1 - alpha) * safe_return + alpha * risky_return  # (n, m, K, Q)
        probabilities = predicted[:, np.newaxis, :, np.newaxis] * self.quadrature_weights  # (n, 1, K, Q)
        beliefs_at_nodes = beliefs[:, np.newaxis, np.newaxis, :]

        if next_age >= RETIREMENT_AGE:
            # Retired next year: V = wealth^(1 - gamma) u(age, belief). Retirees' employment doesn't matter
            next_beliefs = self.update_beliefs(beliefs_at_nodes, self.log_return_nodes[np.newaxis])  # (n, K, Q, K)
            next_wealth = savings_ratio[:, np.newaxis, np.newaxis, np.newaxis] * portfolio_return
            coefficient = self.retired_value_coefficient(next_beliefs[:, np.newaxis], next_age)
            if gamma == 1:
                next_value = self.value_scale(next_age) * np.log(np.maximum(next_wealth, TINY)) + coefficient
            else:
                next_value = np.maximum(next_wealth, TINY) ** (1 - gamma) * coefficient

            return DISCOUNT_FACTOR * np.sum(next_value * probabilities, axis=(2, 3))

        # Still working next year: V = wage^(1 - gamma) v_j'(wealth / wage), where next year's wage is this year's
        # times a growth factor that depends on the employment transition (and the raise)
        expected = 0.0
        for next_employment in range(self.n_employment_states):
            transition = self.employment_transitions[:, employment, next_employment]  # (K,)
            next_beliefs = self.update_beliefs(beliefs_at_nodes, self.log_return_nodes[np.newaxis], employment, next_employment)
            growth, growth_probabilities = self.wage_growth[employment][next_employment]
            next_wealth_ratio = savings_ratio[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * portfolio_return[..., np.newaxis] / growth
            next_value = self.working_value(next_wealth_ratio, next_beliefs[:, np.newaxis, :, :, np.newaxis, :], next_age, next_employment)
            if gamma == 1:
                next_value = self.value_scale(next_age) * np.log(growth) + next_value
            else:
                next_value = growth ** (1 - gamma) * next_value
            next_value = next_value @ growth_probabilities  # (n, m, K, Q)
            expected = expected + np.sum(next_value * probabilities * transition[:, np.newaxis], axis=(2, 3))

        return DISCOUNT_FACTOR * expected

    def choose_portfolio(self, savings_ratio, beliefs, age, employment=EMPLOYED):

        # The risky share that maximizes the expected continuation value, and the maximized value
        # The objective is concave in the risky share, so search a coarse grid, then twice around the best point
        candidates = np.tile(self.risky_share_grid, (savings_ratio.size, 1))
        step = self.risky_share_grid[1] - self.risky_share_grid[0] if self.risky_share_grid[-1] > 0 else 0.0
        upper = self.risky_share_grid[-1]

        for _ in range(3):
            values = self.expected_next_value(savings_ratio, beliefs, candidates, age, employment)
            best_idx = np.argmax(values, axis=1)
            rows = np.arange(savings_ratio.size)
            best_share, best_value = candidates[rows, best_idx], values[rows, best_idx]
            if step == 0:
                break
            candidates = np.clip(best_share[:, np.newaxis] + np.linspace(-step, step, 9), 0.0, upper)
            step /= 4

        return best_share, best_value

    def choose_consumption(self, wealth_ratio, beliefs, age, employment=EMPLOYED):

        # Working ages: the consumption fraction (of wealth + income) that maximizes u(consumption) + post-decision value
        # Everything is in units of the wage. Returns the fractions and the maximized values
        resources = wealth_ratio + self.income_multipliers[employment]

        def best_of(candidates):
            savings_ratio = (1 - candidates) * resources[:, np.newaxis]
            values = self.utility(candidates * resources[:, np.newaxis]) + self.post_decision_value(
                savings_ratio, beliefs[:, np.newaxis, :], age, employment
            )
            rows, best_idx = np.arange(candidates.shape[0]), np.argmax(values, axis=1)
            return candidates[rows, best_idx], values[rows, best_idx]

        candidates = np.tile(CONSUMPTION_FRACTION_GRID, (wealth_ratio.size, 1))
        step = CONSUMPTION_FRACTION_GRID[1] - CONSUMPTION_FRACTION_GRID[0]
        for _ in range(2):
            best_fraction, _ = best_of(candidates)
            candidates = np.clip(best_fraction[:, np.newaxis] + np.linspace(-step, step, 21), MIN_CONSUMPTION_FRACTION, 1.0)
            step /= 10

        return best_of(candidates)

    def retired_policy(self, beliefs, age):

        # Retirees: the consumption fraction (of wealth) and risky share don't depend on wealth
        # Returns (consumption fraction, risky share, value coefficient u(age, belief))
        if age >= MAX_AGE:
            ones = np.ones(beliefs.shape[0])
            return ones, np.full_like(ones, np.nan), self.utility(ones)

        risky_share, post_decision = self.choose_portfolio(np.ones(beliefs.shape[0]), beliefs, age)

        # Closed form consumption: maximize u(f) + (1 - f)^(1 - gamma) * post_decision over f
        if self.gamma == 1:
            fraction = np.full(beliefs.shape[0], 1 / self.value_scale(age))
            value = np.log(fraction) + self.post_decision_scale(age) * np.log(1 - fraction) + post_decision
        else:
            # First order condition: f^(-gamma) = (1 - gamma) * post_decision * (1 - f)^(-gamma)
            fraction = 1 / (1 + ((1 - self.gamma) * post_decision) ** (1 / self.gamma))
            value = self.utility(fraction) + (1 - fraction) ** (1 - self.gamma) * post_decision

        return fraction, risky_share, value

    def solve(self, verbose=True):

        # Backward induction from MAX_AGE
        beliefs = self.belief_grid.beliefs
        belief_shape = self.belief_grid.shape

        for age in reversed(AGE_GRID):
            start = time.time()

            if age >= RETIREMENT_AGE:
                _, _, value = self.retired_policy(beliefs, age)
                self.retired_value[age] = value.reshape(belief_shape)

            else:
                # Post-decision values on (savings ratio, belief), for every employment status. These must all be
                # computed before the values, which depend on them
                savings_mesh = np.repeat(SAVINGS_RATIO_GRID, beliefs.shape[0])
                belief_mesh = np.tile(beliefs, (SAVINGS_RATIO_GRID.size, 1))
                post_decision_tables = []
                for employment in range(self.n_employment_states):
                    _, post_decision = self.choose_portfolio(savings_mesh, belief_mesh, age, employment)
                    post_decision_tables.append(
                        self.invert(post_decision, self.post_decision_scale(age)).reshape((SAVINGS_RATIO_GRID.size,) + belief_shape)
                    )
                self.inverted_post_decision_value[age] = np.stack(post_decision_tables)

                # Values on (wealth ratio, belief)
                wealth_mesh = np.repeat(WEALTH_RATIO_GRID, beliefs.shape[0])
                belief_mesh = np.tile(beliefs, (WEALTH_RATIO_GRID.size, 1))
                value_tables = []
                for employment in range(self.n_employment_states):
                    _, value = self.choose_consumption(wealth_mesh, belief_mesh, age, employment)
                    value_tables.append(self.invert(value, self.value_scale(age)).reshape((WEALTH_RATIO_GRID.size,) + belief_shape))
                self.inverted_value[age] = np.stack(value_tables)

            if verbose and (age % 10 == 0 or age == MAX_AGE):
                print(f"  solved age {age} ({time.time() - start:.1f}s)")

        return self

    def policy(self, wealth, wage, beliefs, age, employment=None):

        # The optimal action at arbitrary states: returns (consumption, risky share), arrays of shape (n,)
        # Vectorized: wealth and wage are arrays of shape (n,), beliefs has shape (n, K), employment is an int array
        # of shape (n,) (everyone is employed by default), age is a scalar
        if age >= RETIREMENT_AGE:
            fraction, risky_share = self.retired_policy(beliefs, age)[:2]
            return fraction * wealth, risky_share

        employment = np.zeros(wealth.size, dtype=int) if employment is None else np.asarray(employment)
        consumption, risky_share = np.zeros(wealth.size), np.zeros(wealth.size)
        for status in np.unique(employment):
            rows = employment == status
            wealth_ratio = wealth[rows] / wage[rows]
            fraction, _ = self.choose_consumption(wealth_ratio, beliefs[rows], age, status)
            resources = wealth_ratio + self.income_multipliers[status]
            risky_share[rows], _ = self.choose_portfolio((1 - fraction) * resources, beliefs[rows], age, status)
            consumption[rows] = fraction * resources * wage[rows]

        return consumption, risky_share


def simulate_lifecycles(model, n_agents, seed):

    # Simulates agents from (INITIAL_WEALTH, INITIAL_INCOME, MIN_AGE) until MAX_AGE. Every agent starts out employed,
    # their initial belief is the stationary distribution of the regime (they haven't seen any returns yet), and
    # their initial regime is drawn from it. Returns a DataFrame with one row per agent per age
    rng = np.random.default_rng(seed)
    hmm = model.hmm

    wealth = np.full(n_agents, INITIAL_WEALTH, dtype=float)
    wage = np.full(n_agents, INITIAL_INCOME, dtype=float)
    employment = np.full(n_agents, EMPLOYED)
    beliefs = np.tile(hmm.initial_distribution, (n_agents, 1))
    regime = rng.choice(hmm.n_regimes, size=n_agents, p=hmm.initial_distribution)
    rows = []

    for age in AGE_GRID:
        working = age < RETIREMENT_AGE
        income = model.income_multipliers[employment] * wage if working else np.zeros(n_agents)
        consumption, risky_share = model.policy(wealth, wage, beliefs, age, employment)
        consumption = np.minimum(consumption, wealth + income)
        record = {
            "agent": np.arange(n_agents),
            "age": age,
            "wealth": wealth,
            "wage": wage if working else np.zeros(n_agents),
            "income": income,
            "unemployed": (employment == UNEMPLOYED) if working else np.zeros(n_agents, dtype=bool),
            "consumption": consumption,
            "consumption_fraction": consumption / (wealth + income),
            "risky_share": risky_share,
            "regime": regime,
        }
        for k, name in enumerate(hmm.regime_names):
            record[f"belief_{name}"] = beliefs[:, k]
        rows.append(pd.DataFrame(record))

        if age == MAX_AGE:
            break

        # Next year's regime, returns, employment status, wage, wealth and belief
        next_regime = (rng.uniform(size=n_agents)[:, np.newaxis] > np.cumsum(hmm.transition_matrix[regime], axis=1)).sum(axis=1)
        log_returns = hmm.simulate_log_returns(next_regime, rng)
        gross_returns = np.exp(log_returns)
        portfolio_return = (1 - risky_share) * gross_returns[:, 0] + risky_share * gross_returns[:, 1]
        wealth = (wealth + income - consumption) * portfolio_return

        if age + 1 >= RETIREMENT_AGE:
            beliefs = model.update_beliefs(beliefs, log_returns)
            employment = np.full(n_agents, EMPLOYED)  # Irrelevant for retirees
        else:
            raise_rate = (rng.uniform(size=n_agents) < RAISE_PROBABILITY) * rng.uniform(MIN_RAISE, MAX_RAISE, size=n_agents)
            next_employment = employment
            if model.n_employment_states > 1:
                transition = model.employment_transitions[next_regime, employment]  # (n_agents, J)
                next_employment = (rng.uniform(size=n_agents)[:, np.newaxis] > np.cumsum(transition, axis=1)).sum(axis=1)
            stays_employed = (employment == EMPLOYED) & (next_employment == EMPLOYED)
            reemployed = (employment == UNEMPLOYED) & (next_employment == EMPLOYED)
            if model.employment is not None:
                wage = np.where(reemployed, wage * model.employment.reemployment_wage_factor, wage)
            wage = np.where(stays_employed, wage * (1 + raise_rate), wage)
            beliefs = model.update_beliefs(beliefs, log_returns, employment, next_employment)
            employment = next_employment
        regime = next_regime

    return pd.concat(rows, ignore_index=True)


def risky_share_by_age(model, savings_ratio_by_age, beliefs, employment=EMPLOYED):

    # The optimal risky share at every age for a few fixed beliefs, holding savings / income at a given level
    # for each working age (retirees' portfolios don't depend on wealth)
    shares = np.zeros((len(beliefs), AGE_GRID.size - 1))
    for age_idx, age in enumerate(AGE_GRID[:-1]):
        savings_ratio = np.full(len(beliefs), savings_ratio_by_age.get(age, 1.0) if age < RETIREMENT_AGE else 1.0)
        shares[:, age_idx], _ = model.choose_portfolio(savings_ratio, beliefs, age, employment)

    return shares


def belief_vector(probability_turbulent):

    return np.array([[1 - p, p] for p in probability_turbulent])


def median_savings_ratio_by_age(df):

    # Savings as a multiple of the wage (which is income, for the employed)
    working = df[df["age"] < RETIREMENT_AGE]
    savings_ratio = (working["wealth"] + working["income"] - working["consumption"]) / working["wage"]
    return savings_ratio.groupby(working["age"]).median().to_dict()


def save_risky_share_by_age_plot(results, outfile="portfolio_hmm_risky_share_by_age.png"):

    # One panel per market: the optimal share of savings in equities by age, for a few fixed beliefs about the regime
    probability_turbulent = [0.02, 0.25, 0.5, 0.75, 0.98]
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 5), sharey=True, squeeze=False)

    for ax, (model, iid_model, df) in zip(axes[0], results):
        savings_ratio = median_savings_ratio_by_age(df)
        shares = risky_share_by_age(model, savings_ratio, belief_vector(probability_turbulent))
        shade_retirement(ax)
        for p, color, share in zip(probability_turbulent, BELIEF_COLORS, shares):
            ax.plot(AGE_GRID[:-1], share, color=color, label=f"Pr(turbulent) = {p:.2f}")

        iid_share = risky_share_by_age(iid_model, savings_ratio, np.ones((1, 1)))[0]
        ax.plot(AGE_GRID[:-1], iid_share, color=IID_COLOR, linestyle="--", label="i.i.d. returns (same long run moments)")

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("age")
        ax.set_title(f"{model.market.name.capitalize()}: share of savings in equities", loc="left")

    axes[0][0].set_ylabel("equity share (savings at the median for each age)")
    axes[0][0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(get_plot_path(outfile))
    plt.close(fig)


def save_lifecycle_plot(results, outfile="portfolio_hmm_lifecycle.png"):

    fig, axes = plt.subplots(len(results), 3, figsize=(17, 4.8 * len(results)), squeeze=False)

    for row, (model, iid_model, df) in zip(axes, results):
        name = model.market.name.capitalize()
        iid_df = simulate_lifecycles(iid_model, n_agents=df["agent"].nunique(), seed=0)

        ax = row[0]
        shade_retirement(ax)
        plot_median_and_band(ax, df, "income", INCOME_COLOR, "income")
        plot_median_and_band(ax, df, "consumption", CALM_COLOR, "consumption (HMM)")
        plot_median_and_band(ax, iid_df, "consumption", IID_COLOR, "consumption (i.i.d.)", "--", band=False)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{name}: income and consumption", loc="left")
        ax.legend(loc="upper left")

        ax = row[1]
        shade_retirement(ax)
        plot_median_and_band(ax, df, "wealth", CALM_COLOR, "wealth (HMM)")
        plot_median_and_band(ax, iid_df, "wealth", IID_COLOR, "wealth (i.i.d.)", "--", band=False)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{name}: wealth", loc="left")
        ax.legend(loc="upper left")

        ax = row[2]
        shade_retirement(ax)
        plot_median_and_band(ax, df, "risky_share", CALM_COLOR, "equity share (HMM)")
        plot_median_and_band(ax, iid_df, "risky_share", IID_COLOR, "equity share (i.i.d.)", "--", band=False)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{name}: share of savings in equities", loc="left")
        ax.legend(loc="lower left")

        for ax in row:
            ax.set_xlabel("age")

    fig.suptitle("Simulated lifecycles (median, 10th-90th percentile band)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(get_plot_path(outfile))
    plt.close(fig)


def save_sample_path_plot(model, df, agent, outfile):

    # One simulated agent: the hidden regime (shaded), the agent's belief, their equity share and their wealth
    path = df[df["agent"] == agent].set_index("age")
    turbulent = path["regime"] == 1

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7), sharex=True)

    for ax in axes:
        for age in path.index[turbulent]:
            ax.axvspan(age - 0.5, age + 0.5, color=TURBULENT_COLOR, alpha=0.15, linewidth=0, zorder=0)
        ax.set_xlim(MIN_AGE - 0.5, MAX_AGE + 0.5)
        ax.axvline(RETIREMENT_AGE - 0.5, color="#898781", linewidth=1, linestyle=":")

    ax = axes[0]
    ax.plot(path.index, path["belief_turbulent"], color=TURBULENT_COLOR, label="belief: Pr(turbulent)")
    ax.plot(path.index, path["risky_share"], color=CALM_COLOR, label="equity share")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{model.market.name.capitalize()}: one simulated agent (shaded years are turbulent)", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    ax = axes[1]
    ax.plot(path.index, path["wealth"], color=CALM_COLOR, label="wealth")
    ax.plot(path.index, path["consumption"], color=IID_COLOR, label="consumption")
    ax.plot(path.index, path["income"], color=INCOME_COLOR, label="income")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("age")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.tight_layout()
    fig.savefig(get_plot_path(outfile))
    plt.close(fig)


def summarize(model, df):

    # Mean equity share and consumption fraction by age group, split by the agent's belief about the regime
    df = df[df["age"] < MAX_AGE].copy()
    df["age group"] = pd.cut(df["age"], bins=[MIN_AGE - 1, 34, 44, 54, 64, 74, MAX_AGE], labels=["25-34", "35-44", "45-54", "55-64", "65-74", "75-84"])
    df["belief"] = pd.cut(df["belief_turbulent"], bins=[-0.01, 0.2, 0.5, 1.01], labels=["Pr(turbulent) < 0.2", "0.2-0.5", "> 0.5"])
    table = df.pivot_table(index="age group", columns="belief", values=["risky_share", "consumption_fraction"], aggfunc="mean", observed=False)
    print(f"\n{model.market.name}: mean equity share and consumption fraction, by age group and belief")
    print(table.round(3).to_string())


def main(n_agents=2000, seed=0):

    plt.rcParams.update(PLOT_STYLE)
    results = []

    for market in [bank_and_equity_market(), bond_and_equity_market()]:
        print(f"Solving: {market.name}")
        model = LifecycleModel(market).solve()
        print(f"Solving: {market.name} (i.i.d. returns)")
        iid_model = LifecycleModel(iid_version(market)).solve()

        df = simulate_lifecycles(model, n_agents, seed)
        summarize(model, df)
        results.append((model, iid_model, df))

    save_risky_share_by_age_plot(results)
    save_lifecycle_plot(results)
    for model, _, df in results:
        slug = "bank" if model.market.asset_names[0] == "bank" else "bonds"
        save_sample_path_plot(model, df, agent=0, outfile=f"portfolio_hmm_sample_path_{slug}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-agents", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(n_agents=args.n_agents, seed=args.seed)
