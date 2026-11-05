import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import RegularGridInterpolator


# State variable: (wealth, income, age)
# Action variable: how much to consume in the current period (implicitly, how much to save for the future)
# No durables like houses, cars, etc
# Savings earn a constant interest rate, no randomness (no equities for simplicity)
# Reward: ln(consumption)
# Income is stochastic: the agent gets random raises (increases in income) until they reach retirement age

# Very simple lifecycle model: the agent lives from age MIN_AGE to MAX_AGE and then dies
# At RETIREMENT_AGE, their income falls to zero
MIN_AGE = 25
RETIREMENT_AGE = 65
MAX_AGE = 85
AGE_GRID = np.arange(MIN_AGE, MAX_AGE + 1)

DISCOUNT_FACTOR = 0.95

# The agent puts their savings in a bank and earns yearly interest
INTEREST_RATE = 0.01

# At age MIN_AGE, the agent earns an income of INITIAL_INCOME
# Every year, they receive a raise with probability RAISE_PROBABILITY
# The agent retires (stops earning income) when they reach RETIREMENT_AGE
INITIAL_INCOME = 1
RAISE_PROBABILITY = 0.5

# If the agent gets a raise, it'll be randomly distributed
MIN_RAISE, MAX_RAISE = (0.0, 0.05)

# The agent begins their career at MIN_AGE with a gift from their parents
INITIAL_WEALTH = 0.1 * INITIAL_INCOME

# The agent's action is represented as the fraction in (0, 1] of (wealth + income) that the agent consumes in the current period
# The agent can't borrow: consuming a fraction of 1.0 leaves them with zero wealth next period
CONSUMPTION_FRACTION_GRID = np.linspace(0.01, 1.0, 100)
MIN_CONSUMPTION_FRACTION = 0.001

# To calculate expected continuation values, we replace the continuous distribution of next year's raise with a
# discrete one: "no raise" with probability 1 - RAISE_PROBABILITY, plus Gauss-Legendre nodes for the uniform raise.
# This computes E[V(next state)] rather than V(E[next state]), which matters because V is concave
_legendre_nodes, _legendre_weights = np.polynomial.legendre.leggauss(5)
RAISE_RATES = np.concatenate([[0.0], MIN_RAISE + (MAX_RAISE - MIN_RAISE) * (_legendre_nodes + 1) / 2])
RAISE_RATE_PROBABILITIES = np.concatenate([[1 - RAISE_PROBABILITY], RAISE_PROBABILITY * _legendre_weights / 2])
EXPECTED_INCOME_GROWTH = np.dot(RAISE_RATE_PROBABILITIES, 1 + RAISE_RATES)

# Exploring starts (Sutton and Barto 5.3): every Monte Carlo episode starts at a random age, wealth, and income.
# Without this, every episode starts from the same state and follows the same deterministic policy, so the visited
# states have almost no variation in wealth at any given age, and the slope of the value fn with respect to wealth
# (which is exactly what the policy update needs) is barely identified
MIN_START_WEALTH, MAX_START_WEALTH = (0.01 * INITIAL_INCOME, 10.0 * INITIAL_INCOME)

# Grids for the backward induction benchmark (denser at low wealth, where the value fn has the most curvature)
BENCHMARK_WEALTH_GRID = INITIAL_INCOME * np.geomspace(1e-3, 20.0, 200)
BENCHMARK_INCOME_GRID = INITIAL_INCOME * np.geomspace(1.0, (1 + MAX_RAISE) ** (RETIREMENT_AGE - 1 - MIN_AGE), 15)

# Floor used before taking logs, so that zero wealth gives a very negative (but finite) value instead of -inf or NaN
TINY = 1e-12

# Ways of extending this problem and making it more interesting:
#  Investment: allow the agent to choose between putting savings in a bank (safe, low interest rate) and in equities (higher expected return, but some volatility)
#   See how optimal investment changes with age -- expect agent to put more of their savings in equities when young
#   (Done in consumption_portfolio_hmm.py, where equity and bond returns follow a hidden Markov model)
#  Home ownership
#  Non-deterministic lifespan :-)
#  Probability of unemployment (downside income risk is what drives precautionary saving, see Carroll's lecture notes)

PLOT_DIR = "plots"

# Plot colors: categorical slots 1-3 of a colorblind-safe palette, plus recessive ink for text, grid and axes
APPROXIMATE_COLOR = "#2a78d6"
BENCHMARK_COLOR = "#eb6834"
INCOME_COLOR = "#1baf7a"
ITERATION_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
RETIRED_SHADE = "#f0efec"
PLOT_STYLE = {
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b",
    "axes.titlesize": 12,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "xtick.labelcolor": "#52514e",
    "ytick.labelcolor": "#52514e",
    "text.color": "#0b0b0b",
    "legend.frameon": False,
    "lines.linewidth": 2,
    "font.size": 11,
}


def reward_function(consumption):

    # The agent's action is how much to consume in the current period
    # They enjoy logarithmic utility from consumption flows
    # (This is the exact same reward function as in the cake eating problem)
    return np.log(consumption)


def get_possible_next_incomes(income, age):

    # The discretized distribution of next period's income (the probabilities are RAISE_RATE_PROBABILITIES)
    # Returns an array of shape (number of states, number of possible raises)
    if age + 1 >= RETIREMENT_AGE:
        return np.zeros((income.size, RAISE_RATES.size))

    return income[:, np.newaxis] * (1 + RAISE_RATES)


def simulate_next_state(wealth, income, age, consumption, rng):

    # Vectorized: wealth, income and consumption are arrays (one element per agent), age is a scalar
    # This state variable tracks the agent's wealth at the _beginning_ of each period
    next_wealth = (wealth + income - consumption) * (1 + INTEREST_RATE)

    if age + 1 >= RETIREMENT_AGE:
        return next_wealth, np.zeros_like(income)

    agent_receives_raise = rng.uniform(size=income.size) < RAISE_PROBABILITY
    rate = rng.uniform(low=MIN_RAISE, high=MAX_RAISE, size=income.size)
    next_income = income * (1 + agent_receives_raise * rate)

    return next_wealth, next_income


def human_wealth_factor(age):

    # Human wealth (Carroll's term) is the expected present discounted value of future labor income
    # This is the ratio of human wealth to current income: the sum over remaining working years k = 1, ..., K
    # of (EXPECTED_INCOME_GROWTH / (1 + INTEREST_RATE)) ** k. It's zero in the last working year and in retirement
    remaining_working_years = np.clip(RETIREMENT_AGE - 1 - np.asarray(age), 0, None)
    k = np.arange(1, RETIREMENT_AGE - MIN_AGE + 1)
    discounted_growth = (EXPECTED_INCOME_GROWTH / (1 + INTEREST_RATE)) ** k

    return np.sum(discounted_growth * (k <= remaining_working_years[..., np.newaxis]), axis=-1)


def log_total_wealth(wealth, income, age):

    # Total wealth = wealth + this period's income + human wealth (the expected value of all future income)
    total_wealth = wealth + income * (1 + human_wealth_factor(age))

    return np.log(np.maximum(total_wealth, TINY))


def get_feature_vector(wealth, income, age):

    # Sutton and Barto 9.4: "the vector x(s) is called a feature vector representing state s"
    # Vectorized: wealth, income and age are arrays of the same shape, and the features are in a new last dimension

    # The approximate value fn is V(s) = constant[age] + slope[age] * ln(total wealth)
    # Without income uncertainty or a binding borrowing constraint, this functional form is exact (it's a finite horizon
    # cake eating problem where the cake is total wealth, see Carroll's PerfForesightCRRA lecture notes)
    # Total wealth includes future income: a young worker with no savings is much richer than ln(wealth + income) suggests
    age_one_hot = 1.0 * (np.asarray(age)[..., np.newaxis] == AGE_GRID)
    log_total_wealth_age_interaction = log_total_wealth(wealth, income, age)[..., np.newaxis] * age_one_hot

    return np.concatenate([age_one_hot, log_total_wealth_age_interaction], axis=-1)


def approximate_value(value_fn_coefficients, wealth, income, age):

    # Same as np.dot(get_feature_vector(wealth, income, age), value_fn_coefficients), but cheaper when age is a scalar:
    # only that age's constant and ln(total wealth) coefficient are nonzero in the dot product
    age_idx = age - MIN_AGE
    constant = value_fn_coefficients[age_idx]
    slope = value_fn_coefficients[AGE_GRID.size + age_idx]

    return constant + slope * log_total_wealth(wealth, income, age)


def remaining_lifetime_discount_sum(age):

    # 1 + beta + beta^2 + ..., with one term per remaining year of life (including the current year)
    # This is the coefficient on ln(total wealth) in the perfect foresight value fn
    return np.sum(DISCOUNT_FACTOR ** np.arange(MAX_AGE - age + 1))


def perfect_foresight_value_fn_coefficients():

    # Closed form value fn coefficients (constant and slope on ln(total wealth) at every age) for an agent
    # who faces no income uncertainty. Exact at every age in retirement, since there's no income left to be uncertain about
    # Backward recursion from the final age, where the agent consumes everything and V = ln(wealth)
    slopes = np.array([remaining_lifetime_discount_sum(age) for age in AGE_GRID])
    constants = np.zeros(AGE_GRID.size)
    for age_idx in reversed(range(AGE_GRID.size - 1)):
        next_slope = slopes[age_idx + 1]

        # Optimal consumption is total wealth / slopes[age_idx]: plug that into the Bellman equation
        constants[age_idx] = (
            -np.log(slopes[age_idx])
            + DISCOUNT_FACTOR * next_slope * np.log((1 + INTEREST_RATE) * DISCOUNT_FACTOR * next_slope / slopes[age_idx])
            + DISCOUNT_FACTOR * constants[age_idx + 1]
        )

    return np.concatenate([constants, slopes])


def choose_consumption_fraction(wealth, income, age, value_fn):

    # Returns the consumption fractions that maximize reward + DISCOUNT_FACTOR * E[value_fn(next state)],
    # along with the maximized values. Vectorized: wealth and income are 1d arrays, age is a scalar
    # value_fn(wealth, income, age) is either the approximate value fn or the backward induction benchmark

    # Shortcut: optimal policy in final year is to consume everything
    if age >= MAX_AGE:
        return np.ones_like(wealth), reward_function(wealth + income)

    resources = wealth + income
    next_income = get_possible_next_incomes(income, age)

    def value_of(fractions):

        # Fractions has shape (number of states, number of candidate fractions)
        consumption = fractions * resources[:, np.newaxis]
        next_wealth = (resources[:, np.newaxis] - consumption) * (1 + INTEREST_RATE)

        # Shape (number of states, number of candidate fractions, number of possible raises)
        next_value = value_fn(next_wealth[:, :, np.newaxis], next_income[:, np.newaxis, :], age + 1)
        expected_next_value = np.dot(next_value, RAISE_RATE_PROBABILITIES)

        return reward_function(consumption) + DISCOUNT_FACTOR * expected_next_value

    def best_of(candidate_fractions):

        values = value_of(candidate_fractions)
        rows = np.arange(candidate_fractions.shape[0])
        best_idx = np.argmax(values, axis=1)

        return candidate_fractions[rows, best_idx], values[rows, best_idx]

    # Brute force search over a coarse grid, then twice over a 10x finer grid around the best candidate so far
    # (the objective is concave in consumption, so this finds the maximum to within 0.0001)
    candidates = np.tile(CONSUMPTION_FRACTION_GRID, (wealth.size, 1))
    step = CONSUMPTION_FRACTION_GRID[1] - CONSUMPTION_FRACTION_GRID[0]
    for _ in range(2):
        best_fraction, _ = best_of(candidates)
        candidates = np.clip(
            best_fraction[:, np.newaxis] + np.linspace(-step, step, 21), MIN_CONSUMPTION_FRACTION, 1.0
        )
        step /= 10

    return best_of(candidates)


def get_greedy_policy(value_fn):

    # Policy improvement (Sutton and Barto 4.2): act greedily with respect to a value fn, using the known transition model
    # A policy maps (wealth, income, age) to the fraction of wealth + income consumed in the current period
    def policy(wealth, income, age):
        consumption_fraction, _ = choose_consumption_fraction(wealth, income, age, value_fn)
        return consumption_fraction

    return policy


def initial_policy(wealth, income, age):

    # Consume the smallest possible fraction of wealth + income at every state, including at MAX_AGE
    return np.min(CONSUMPTION_FRACTION_GRID) * np.ones_like(wealth)


def simulate_episodes(policy, start_wealth, start_income, start_age, rng):

    # Simulates every episode (vectorized across episodes) from its start state until the agent dies at MAX_AGE
    # Returns arrays of wealth, income and consumption, with shape (number of episodes, number of ages)
    # and NaN at ages before an episode starts
    n_episodes = start_age.size
    wealth, income, consumption = (np.full((n_episodes, AGE_GRID.size), np.nan) for _ in range(3))

    current_wealth, current_income = start_wealth.astype(float), start_income.astype(float)

    for age_idx, age in enumerate(AGE_GRID):

        idx = np.flatnonzero(start_age <= age)
        wealth[idx, age_idx] = current_wealth[idx]
        income[idx, age_idx] = current_income[idx]

        # Convention: the policy tells us the _fraction_ of (wealth + income) that the agent will consume in the current period
        consumption_fraction = policy(current_wealth[idx], current_income[idx], age)
        consumption[idx, age_idx] = consumption_fraction * (current_wealth[idx] + current_income[idx])

        if age < MAX_AGE:
            current_wealth[idx], current_income[idx] = simulate_next_state(
                current_wealth[idx], current_income[idx], age, consumption[idx, age_idx], rng
            )

    return wealth, income, consumption


def calculate_returns(rewards):

    # The return G_t is the discounted sum of rewards from age t until the end of the episode (Sutton and Barto 3.3)
    # Computed backwards, since G_t = R_t + DISCOUNT_FACTOR * G_{t+1}
    returns = np.zeros_like(rewards)
    future_return = np.zeros(rewards.shape[0])

    for age_idx in reversed(range(AGE_GRID.size)):
        future_return = rewards[:, age_idx] + DISCOUNT_FACTOR * future_return
        returns[:, age_idx] = future_return

    return returns


def calculate_approx_value_function(policy, rng, n_episodes=4000):

    # Monte Carlo policy evaluation with linear function approximation: simulate episodes under the policy,
    # then regress the observed returns on the state features (Sutton and Barto 5.1)
    # Least squares on the whole batch of episodes is the fit that Sutton and Barto 9.3's gradient Monte Carlo
    # (one SGD step per visited state) converges to, but SGD gets there slowly when features are nearly collinear

    # Exploring starts: random start age, wealth, and income (an income that's reachable by that age)
    start_age = rng.choice(AGE_GRID, size=n_episodes)
    start_wealth = np.exp(rng.uniform(np.log(MIN_START_WEALTH), np.log(MAX_START_WEALTH), size=n_episodes))
    years_of_raises = np.clip(start_age - MIN_AGE, 0, RETIREMENT_AGE - 1 - MIN_AGE)
    max_income = INITIAL_INCOME * (1 + MAX_RAISE) ** years_of_raises
    start_income = np.where(start_age < RETIREMENT_AGE, rng.uniform(INITIAL_INCOME, max_income), 0.0)

    wealth, income, consumption = simulate_episodes(policy, start_wealth, start_income, start_age, rng)
    returns = calculate_returns(reward_function(consumption))

    visited = ~np.isnan(wealth)
    ages = np.broadcast_to(AGE_GRID, wealth.shape)
    features = get_feature_vector(wealth[visited], income[visited], ages[visited])

    value_fn_coefficients, *_ = np.linalg.lstsq(features, returns[visited], rcond=None)

    return value_fn_coefficients


def solve_by_backward_induction():

    # Benchmark solution: dynamic programming by backward induction, starting from MAX_AGE, on a grid of wealth and income
    # This doesn't assume any functional form for the value fn (it interpolates between grid points), so we can use it
    # to check the approximate policy iteration solution (similar to how the cake eating problem has an exact solution)
    # Returns a value fn with the same signature as the approximate value fn: value_fn(wealth, income, age)
    #
    # Rather than interpolating V itself (which is very concave in wealth, so linear interpolation
    # gets its slope wrong, and the slope is what determines how much the agent saves), we interpolate
    # the "inverted" value fn exp(V / (1 + beta + beta^2 + ...)), which is close to linear in wealth and income
    # (exactly linear in total wealth when there's no income uncertainty). This is the same trick as in
    # Carroll's "Solution Methods for Microeconomic Dynamic Stochastic Optimization Problems"
    interpolators = {}

    def value_fn(wealth, income, age):
        wealth, income = np.broadcast_arrays(wealth, income)
        if age >= RETIREMENT_AGE:
            # Retired agents have no income, so their value only depends on wealth
            inverted_value = interpolators[age](wealth[..., np.newaxis])
        else:
            inverted_value = interpolators[age](np.stack([wealth, income], axis=-1))

        return remaining_lifetime_discount_sum(age) * np.log(np.maximum(inverted_value, TINY))

    for age in reversed(AGE_GRID):

        if age >= RETIREMENT_AGE:
            wealth = BENCHMARK_WEALTH_GRID
            income = np.zeros_like(wealth)
            points = (BENCHMARK_WEALTH_GRID,)
        else:
            wealth_mesh, income_mesh = np.meshgrid(BENCHMARK_WEALTH_GRID, BENCHMARK_INCOME_GRID, indexing="ij")
            wealth, income = wealth_mesh.ravel(), income_mesh.ravel()
            points = (BENCHMARK_WEALTH_GRID, BENCHMARK_INCOME_GRID)

        _, values = choose_consumption_fraction(wealth, income, age, value_fn)
        inverted_values = np.exp(values / remaining_lifetime_discount_sum(age))

        # fill_value=None means linear extrapolation outside of the grid
        interpolators[age] = RegularGridInterpolator(
            points=points,
            values=inverted_values.reshape([len(p) for p in points]),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    return value_fn


def simulate_lifecycles(policy, n_agents, seed):

    # Simulates agents from (INITIAL_WEALTH, INITIAL_INCOME, MIN_AGE) until MAX_AGE
    # The seed is fixed so that every policy we simulate faces the same random raises (common random numbers)
    rng = np.random.default_rng(seed)
    start_age = np.full(n_agents, MIN_AGE)
    wealth, income, consumption = simulate_episodes(
        policy, np.full(n_agents, INITIAL_WEALTH), np.full(n_agents, INITIAL_INCOME), start_age, rng
    )
    returns = calculate_returns(reward_function(consumption))

    # We construct a DataFrame (one row per agent per age) so that we can plot it later
    return pd.DataFrame(
        {
            "agent": np.repeat(np.arange(n_agents), AGE_GRID.size),
            "age": np.tile(AGE_GRID, n_agents),
            "wealth": wealth.ravel(),
            "income": income.ravel(),
            "consumption": consumption.ravel(),
            "consumption_fraction": (consumption / (wealth + income)).ravel(),
            "discounted_return": returns.ravel(),
        }
    )


def get_plot_path(outfile):

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOT_DIR, outfile)


def shade_retirement(ax):

    ax.axvspan(RETIREMENT_AGE - 0.5, MAX_AGE + 0.5, color=RETIRED_SHADE, zorder=0, linewidth=0)
    ax.text(RETIREMENT_AGE - 0.3, 0.97, "retired", transform=ax.get_xaxis_transform(), va="top", color="#52514e")
    ax.set_xlim(MIN_AGE - 0.5, MAX_AGE + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))


def plot_median_and_band(ax, df, column, color, label, linestyle="-", band=True):

    quantiles = df.groupby("age")[column].quantile([0.1, 0.5, 0.9]).unstack()
    ax.plot(quantiles.index, quantiles[0.5], color=color, linestyle=linestyle, label=label)
    if band:
        ax.fill_between(quantiles.index, quantiles[0.1], quantiles[0.9], color=color, alpha=0.15, linewidth=0)


def save_lifecycle_plot(approximate_df, benchmark_df):

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    shade_retirement(ax)
    plot_median_and_band(ax, approximate_df, "income", INCOME_COLOR, "income")
    plot_median_and_band(ax, approximate_df, "consumption", APPROXIMATE_COLOR, "consumption (approximate policy iteration)")
    plot_median_and_band(
        ax, benchmark_df, "consumption", BENCHMARK_COLOR, "consumption (backward induction benchmark)", "--", band=False
    )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("age")
    ax.set_ylabel("income and consumption per year")
    ax.set_title("Income and consumption (median, 10th-90th percentile band)", loc="left")
    ax.legend(loc="lower left")

    ax = axes[1]
    shade_retirement(ax)
    plot_median_and_band(ax, approximate_df, "wealth", APPROXIMATE_COLOR, "approximate policy iteration")
    plot_median_and_band(ax, benchmark_df, "wealth", BENCHMARK_COLOR, "backward induction benchmark", "--", band=False)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("age")
    ax.set_ylabel("wealth at the beginning of the year")
    ax.set_title("Wealth (median, 10th-90th percentile band)", loc="left")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(get_plot_path("consumption_savings_lifecycle.png"))
    plt.close(fig)


def save_policy_plot(approximate_policy, benchmark_policy, approximate_df):

    # Amount consumed as a function of wealth (the policy), at a few ages, holding income at its median for that age
    ages = [MIN_AGE, RETIREMENT_AGE - 1, MAX_AGE - 2]
    wealth = np.linspace(0.01, 5.0, 200) * INITIAL_INCOME
    median_income = approximate_df.groupby("age")["income"].median()

    fig, axes = plt.subplots(1, len(ages), figsize=(13, 4.5), sharey=True)

    for ax, age in zip(axes, ages):
        income = median_income[age] * np.ones_like(wealth)
        for policy, color, linestyle, label in [
            (approximate_policy, APPROXIMATE_COLOR, "-", "approximate policy iteration"),
            (benchmark_policy, BENCHMARK_COLOR, "--", "backward induction benchmark"),
        ]:
            ax.plot(wealth, policy(wealth, income, age) * (wealth + income), color=color, linestyle=linestyle, label=label)

        ax.set_title(f"age {age}, income {income[0]:.2f}", loc="left")
        ax.set_xlabel("wealth (part of the state variable)")

    axes[0].set_ylabel("amount consumed in current period (action)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(get_plot_path("consumption_savings_policy.png"))
    plt.close(fig)


def save_value_fn_coefficients_plot(value_fn_coefficients):

    # Sanity check: compare the estimated value fn coefficients to the perfect foresight closed form,
    # which is exact for retired agents and very close for working agents (whose raises are small)
    closed_form = perfect_foresight_value_fn_coefficients()
    n_ages = AGE_GRID.size

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, idx, title in [
        (axes[0], slice(n_ages, 2 * n_ages), "Coefficient on ln(total wealth), by age"),
        (axes[1], slice(0, n_ages), "Constant, by age"),
    ]:
        shade_retirement(ax)
        ax.plot(AGE_GRID, closed_form[idx], color=BENCHMARK_COLOR, linestyle="--", label="perfect foresight closed form")
        ax.plot(
            AGE_GRID,
            value_fn_coefficients[idx],
            "o",
            color=APPROXIMATE_COLOR,
            markersize=7,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1.5,
            label="Monte Carlo estimate (approximate policy iteration)",
        )
        ax.set_xlabel("age")
        ax.set_title(title, loc="left")

    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(get_plot_path("consumption_savings_value_fn_coefficients.png"))
    plt.close(fig)


def save_policy_iteration_plot(lifecycle_dfs, benchmark_df):

    # Average consumption (from the initial state) of the policy evaluated at each iteration of approximate policy iteration
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    shade_retirement(ax)

    for policy_update_idx, df in enumerate(lifecycle_dfs):
        mean_consumption = df.groupby("age")["consumption"].mean()
        color = ITERATION_COLORS[min(policy_update_idx, len(ITERATION_COLORS) - 1)]
        label = f"policy {policy_update_idx}" + (" (initial policy)" if policy_update_idx == 0 else "")
        ax.plot(mean_consumption.index, mean_consumption, color=color, label=label)

    benchmark_mean_consumption = benchmark_df.groupby("age")["consumption"].mean()
    ax.plot(
        benchmark_mean_consumption.index,
        benchmark_mean_consumption,
        color=BENCHMARK_COLOR,
        linestyle="--",
        label="backward induction benchmark",
    )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("age")
    ax.set_ylabel("mean consumption")
    ax.set_title("Consumption under each policy during approximate policy iteration", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.tight_layout()
    fig.savefig(get_plot_path("consumption_savings_policy_iteration.png"))
    plt.close(fig)


def main(n_policy_updates=5, n_episodes=4000, n_lifecycles=2000, seed=0):

    plt.rcParams.update(PLOT_STYLE)
    rng = np.random.default_rng(seed)

    print("Solving by backward induction (benchmark)")
    benchmark_value_fn = solve_by_backward_induction()
    benchmark_policy = get_greedy_policy(benchmark_value_fn)
    benchmark_df = simulate_lifecycles(benchmark_policy, n_lifecycles, seed)
    benchmark_return = benchmark_df.loc[benchmark_df["age"] == MIN_AGE, "discounted_return"].mean()

    # Approximate policy iteration: evaluate the current policy by Monte Carlo (fitting an approximate value fn),
    # then update the policy by acting greedily with respect to the approximate value fn
    policy = initial_policy
    lifecycle_dfs = []
    previous_coefficients = None

    for policy_update_idx in range(n_policy_updates):

        value_fn_coefficients = calculate_approx_value_function(policy, rng, n_episodes)

        df = simulate_lifecycles(policy, n_lifecycles, seed)
        lifecycle_dfs.append(df)
        policy_return = df.loc[df["age"] == MIN_AGE, "discounted_return"].mean()

        change = np.nan if previous_coefficients is None else np.max(np.abs(value_fn_coefficients - previous_coefficients))
        print(
            f"Policy {policy_update_idx}: mean discounted lifetime utility {policy_return:.4f} "
            f"(benchmark {benchmark_return:.4f}), max change in value fn coefficients {change:.4f}"
        )
        previous_coefficients = value_fn_coefficients

        policy = get_greedy_policy(
            lambda wealth, income, age, coefficients=value_fn_coefficients: approximate_value(coefficients, wealth, income, age)
        )

    approximate_df = simulate_lifecycles(policy, n_lifecycles, seed)
    approximate_return = approximate_df.loc[approximate_df["age"] == MIN_AGE, "discounted_return"].mean()
    print(f"Final policy: mean discounted lifetime utility {approximate_return:.4f} (benchmark {benchmark_return:.4f})")

    summary = pd.DataFrame(
        {
            "median income": approximate_df.groupby("age")["income"].median(),
            "median consumption": approximate_df.groupby("age")["consumption"].median(),
            "median consumption (benchmark)": benchmark_df.groupby("age")["consumption"].median(),
            "median wealth": approximate_df.groupby("age")["wealth"].median(),
            "median wealth (benchmark)": benchmark_df.groupby("age")["wealth"].median(),
        }
    )
    print(summary.round(3).to_string())

    save_lifecycle_plot(approximate_df, benchmark_df)
    save_policy_plot(policy, benchmark_policy, approximate_df)
    save_value_fn_coefficients_plot(value_fn_coefficients)
    save_policy_iteration_plot(lifecycle_dfs + [approximate_df], benchmark_df)


if __name__ == "__main__":
    main()
