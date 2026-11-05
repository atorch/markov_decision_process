"""How does unemployment risk change the lifecycle portfolio, and does it matter if job losses cluster in turbulent markets?

Solves the lifecycle consumption + portfolio model in consumption_portfolio_hmm.py under five scenarios:

* no unemployment risk (the model in consumption_portfolio_hmm.py)
* moderate unemployment risk (about 5% unemployment in calm years and 9% in turbulent years), either higher in
  turbulent years or independent of the regime
* severe unemployment risk (about 4% unemployment in calm years, 21% in turbulent years, and a bigger wage scar when
  the worker finds a new job), again either higher in turbulent years or independent of the regime

The "independent of the regime" version of each scenario uses the same transition probabilities in every regime,
calibrated so that the long run unemployment rate and the frequency of job losses are the same as in the "higher in
turbulent years" version (see EmploymentRisk.regime_independent). Comparing the two isolates the effect of the
correlation between job losses and equity returns, holding the amount of unemployment risk fixed.
"""

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consumption_portfolio_hmm import (
    AGE_GRID,
    CALM_COLOR,
    IID_COLOR,
    MAX_AGE,
    MIN_AGE,
    RETIREMENT_AGE,
    TURBULENT_COLOR,
    EmploymentRisk,
    LifecycleModel,
    bank_and_equity_market,
    belief_vector,
    bond_and_equity_market,
    median_savings_ratio_by_age,
    regime_dependent_employment_risk,
    risky_share_by_age,
    simulate_lifecycles,
)
from consumption_savings import PLOT_STYLE, get_plot_path, shade_retirement

# Severe unemployment risk: job losses are rare in calm years and very common in turbulent years (a depression), jobs
# are hard to find in turbulent years, and workers who find a new job are paid 25% less than in their old job
SEVERE_SEPARATION_PROBABILITIES = (0.02, 0.12)
SEVERE_JOB_FINDING_PROBABILITIES = (0.65, 0.35)
SEVERE_REEMPLOYMENT_WAGE_FACTOR = 0.75

# Beliefs (Pr(turbulent)) at which to compare policies, and the ages to plot (retirees' policies are the same in every
# scenario: once retired, employment risk is gone)
PROBABILITY_TURBULENT = [0.1, 0.5, 0.9]
MAX_PLOT_AGE = 72

AGE_GROUPS = ([MIN_AGE - 1, 34, 44, 54, 64, MAX_AGE - 1], ["25-34", "35-44", "45-54", "55-64", "65-84"])


@dataclass
class Scenario:

    name: str
    employment: EmploymentRisk | None
    color: str
    linestyle: str


def get_scenarios(market):

    moderate = regime_dependent_employment_risk()
    severe = EmploymentRisk(
        SEVERE_SEPARATION_PROBABILITIES, SEVERE_JOB_FINDING_PROBABILITIES, reemployment_wage_factor=SEVERE_REEMPLOYMENT_WAGE_FACTOR
    )

    # Hue: does unemployment risk depend on the regime? Line style: how much unemployment risk is there?
    return [
        Scenario("no unemployment risk", None, IID_COLOR, "-"),
        Scenario("moderate, independent of regime", moderate.regime_independent(market.hmm), CALM_COLOR, "--"),
        Scenario("moderate, higher when turbulent", moderate, TURBULENT_COLOR, "--"),
        Scenario("severe, independent of regime", severe.regime_independent(market.hmm), CALM_COLOR, "-"),
        Scenario("severe, higher when turbulent", severe, TURBULENT_COLOR, "-"),
    ]


def glide_path_start(shares):

    # The first age at which the equity share falls below 99%, for each row of shares (one row per belief)
    return [int(AGE_GRID[np.argmax(share < 0.99)]) for share in shares]


def summarize(market, results, savings_ratio):

    beliefs = belief_vector(PROBABILITY_TURBULENT)
    rows = []
    for scenario, model, df in results:
        working = df[df["age"] < RETIREMENT_AGE]
        not_dead = df[df["age"] < MAX_AGE]
        shares = risky_share_by_age(model, savings_ratio, beliefs)
        row = {"scenario": scenario.name}
        for p, start in zip(PROBABILITY_TURBULENT, glide_path_start(shares)):
            row[f"glide path starts, Pr(turb) = {p}"] = start
        for regime, name in enumerate(model.hmm.regime_names):
            row[f"unemployment rate ({name})"] = working.loc[working["regime"] == regime, "unemployed"].mean()
        for age in [40, 50, 64]:
            row[f"median wealth / initial income, {age}"] = df.loc[df["age"] == age, "wealth"].median()
        groups = pd.cut(not_dead["age"], bins=AGE_GROUPS[0], labels=AGE_GROUPS[1])
        for group, share in not_dead["risky_share"].groupby(groups, observed=False).mean().items():
            row[f"mean equity share, {group}"] = share
        for unemployed, label in [(False, "employed"), (True, "unemployed")]:
            subset = working[working["unemployed"] == unemployed]
            row[f"mean Pr(turb), {label} workers"] = subset["belief_turbulent"].mean() if len(subset) else np.nan
            row[f"mean equity share, {label} workers"] = subset["risky_share"].mean() if len(subset) else np.nan
        rows.append(row)

    table = pd.DataFrame(rows).set_index("scenario").T
    print(f"\n{market.name}")
    print(table.round(3).to_string())

    return table


def save_risky_share_by_age_plot(all_results, outfile="portfolio_unemployment_risky_share_by_age.png"):

    # Rows: markets. Columns: beliefs. Lines: scenarios. Employed workers, with savings / wage held at the median
    # for each age in the no unemployment risk scenario, so the lines differ only because the policy differs
    ages = AGE_GRID[:-1]
    fig, axes = plt.subplots(len(all_results), len(PROBABILITY_TURBULENT), figsize=(17, 4.8 * len(all_results)), sharey=True, squeeze=False)

    for row, (market, results, savings_ratio) in zip(axes, all_results):
        for ax, p in zip(row, PROBABILITY_TURBULENT):
            shade_retirement(ax)
            for scenario, model, _ in results:
                share = risky_share_by_age(model, savings_ratio, belief_vector([p]))[0]
                ax.plot(ages, share, color=scenario.color, linestyle=scenario.linestyle, label=scenario.name)
            ax.set_xlim(MIN_AGE, MAX_PLOT_AGE)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{market.name.capitalize()}, Pr(turbulent) = {p}", loc="left")
            ax.set_xlabel("age")
        row[0].set_ylabel("equity share (employed worker)")

    axes[0][0].legend(loc="lower left")
    fig.suptitle(
        "Optimal share of savings in equities, holding savings at the no-unemployment-risk median for each age",
        x=0.01,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(get_plot_path(outfile))
    plt.close(fig)


def save_lifecycle_plot(all_results, outfile="portfolio_unemployment_lifecycle.png"):

    # Simulated lifecycles. Columns: median wealth, mean equity share, and (for the scenarios where unemployment
    # is higher in turbulent years) the mean belief and the mean equity share of employed vs unemployed workers
    fig, axes = plt.subplots(len(all_results), 4, figsize=(22, 4.8 * len(all_results)), squeeze=False)

    for row, (market, results, _) in zip(axes, all_results):
        name = market.name.capitalize()
        for scenario, _, df in results:
            by_age = df[df["age"] < MAX_AGE].groupby("age")
            style = {"color": scenario.color, "linestyle": scenario.linestyle, "label": scenario.name}
            row[0].plot(by_age["wealth"].median(), **style)
            row[1].plot(by_age["risky_share"].mean(), **style)

            if scenario.employment is not None and np.ptp(scenario.employment.separation_probabilities) > 0:
                working = df[df["age"] < RETIREMENT_AGE]
                severity = scenario.name.split(",")[0]
                for unemployed, color in [(False, CALM_COLOR), (True, TURBULENT_COLOR)]:
                    subset = working[working["unemployed"] == unemployed].groupby("age")
                    label = f"{severity}: {'unemployed' if unemployed else 'employed'}"
                    row[2].plot(subset["belief_turbulent"].mean(), color=color, linestyle=scenario.linestyle, label=label)
                    row[3].plot(subset["risky_share"].mean(), color=color, linestyle=scenario.linestyle, label=label)

        row[0].set_title(f"{name}: median wealth", loc="left")
        row[0].set_ylim(bottom=0)
        row[1].set_title(f"{name}: mean equity share", loc="left")
        row[1].set_ylim(-0.02, 1.02)
        row[2].set_title(f"{name}: mean Pr(turbulent), workers", loc="left")
        row[2].set_ylim(0, 1)
        row[3].set_title(f"{name}: mean equity share, workers", loc="left")
        row[3].set_ylim(-0.02, 1.02)
        for ax in row:
            shade_retirement(ax)
            ax.set_xlabel("age")
        for ax in row[2:]:
            ax.set_xlim(MIN_AGE, RETIREMENT_AGE)

    axes[0][0].legend(loc="upper left")
    axes[0][2].legend(loc="upper left")
    fig.suptitle(
        "Simulated lifecycles. Right two panels: scenarios where unemployment is higher in turbulent years, employed vs unemployed workers",
        x=0.01,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(get_plot_path(outfile))
    plt.close(fig)


def main(n_agents=2000, seed=0, markets=("bank", "bonds")):

    plt.rcParams.update(PLOT_STYLE)
    all_results = []

    for market in [bank_and_equity_market(), bond_and_equity_market()]:
        if market.asset_names[0] not in markets:
            continue
        results = []
        for scenario in get_scenarios(market):
            print(f"Solving: {market.name}, {scenario.name}", flush=True)
            model = LifecycleModel(market, employment=scenario.employment).solve(verbose=False)
            results.append((scenario, model, simulate_lifecycles(model, n_agents, seed)))

        savings_ratio = median_savings_ratio_by_age(results[0][2])
        summarize(market, results, savings_ratio)
        all_results.append((market, results, savings_ratio))

    save_risky_share_by_age_plot(all_results)
    save_lifecycle_plot(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-agents", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markets", nargs="+", default=["bank", "bonds"], choices=["bank", "bonds"])
    args = parser.parse_args()
    main(n_agents=args.n_agents, seed=args.seed, markets=args.markets)
