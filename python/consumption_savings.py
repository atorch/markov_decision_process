import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


# State variable: (wealth, income, age)
# Action variable: how much to consume (implicitly, how much to save)
# No durables like houses, cars, etc
# Savings earn a constant interest rate, no randomness (no equities for simplicity)
# Reward: utility(consumption), pick something simple like CRRA
# Income is stochastic: you might randomly get a raise (or get fired)


# Very simple lifecycle model: the agent lives from age MIN_AGE to MAX_AGE and then dies
MIN_AGE = 30
RETIREMENT_AGE = 50
MAX_AGE = 60
AGE_GRID = np.arange(MIN_AGE, MAX_AGE + 1)

DISCOUNT_FACTOR = 0.95

# The agent puts their savings in a bank and earns yearly interest
# TODO Make this random?  But would have to be careful calculating E[log(wealth)] in continuation values
INTEREST_RATE = 0.01

# At age MIN_AGE, the agent earns an income of INITIAL_INCOME
# Every year, they receive a raise with probability RAISE_PROBABILITY
# The agent retires (stops earning income) when they reach RETIREMENT_AGE
INITIAL_INCOME = 1
RAISE_PROBABILITY = 0.5

# If the agent gets a raise, it'll be randomly distributed
MIN_RAISE, MAX_RAISE = (0.0, 0.05)

# The agent begins their career at MIN_AGE with a little gift from their parents
INITIAL_WEALTH = INITIAL_INCOME

CONSUMPTION_FRACTION_GRID = np.arange(0.01, 1.0, 0.01)

INCOME_GRID = np.hstack([np.zeros((1, )), INITIAL_INCOME * (1 + MAX_RAISE) ** np.arange(RETIREMENT_AGE - MIN_AGE + 1)])

# TODO Calculate max possible wealth, choose a better grid
WEALTH_GRID = np.arange(0.001 * INITIAL_INCOME, 0.3 * (RETIREMENT_AGE - MIN_AGE) * np.max(INCOME_GRID), 0.1 * INITIAL_INCOME)

# Ways of extending this problem and making it more interesting:
#  Investment: allow the agent to choose between putting savings in a bank (safe, low interest rate) and in equities (higher expected return, but some volatility)
#   See how optimal investment changes with age -- expect agent to put more of their savings in equities when young
#  Home ownership
#  Non-deterministic lifespan :-)
#  Probability of unemployment

PLOT_DIR = "plots"

def reward_function(consumption):

    # The agent's action is how much to consume in the current period
    # They enjoy logarithmic utility from consumption flows
    return np.log(consumption)


def is_terminal_state(wealth, income, age):

    # The problem ends (the agent dies) when they reach the max age
    return age >= MAX_AGE


def simulate_next_state(wealth, income, age, consumption):

    next_age = age + 1
    next_wealth = (wealth + income - consumption) * (1 + INTEREST_RATE)

    if next_age >= RETIREMENT_AGE:
        next_income = 0
        return next_wealth, next_income, next_age

    next_income = income

    agent_receives_raise = np.random.uniform() < RAISE_PROBABILITY
    if agent_receives_raise:
        rate = np.random.uniform(low=MIN_RAISE, high=MAX_RAISE)
        next_income = income * (1 + rate)

    return next_wealth, next_income, next_age


def get_feature_vector(wealth, income, age):

    # Sutton and Barto 9.4: "the vector x(s) is called a feature vector representing state s"

    # TODO Could include interactions, higher order terms, etc
    # TODO Partial dependence plots for approximate value fn
    age_scaled = age / 100
    return np.array([
        1.0,
        np.log(wealth),
        income,
        age_scaled,
        np.log(wealth) * income,        
        np.log(wealth) * age_scaled,
        income * age_scaled,
    ])


def feature_vector_size():

    return get_feature_vector(INITIAL_WEALTH, INITIAL_INCOME, MIN_AGE).size


def discounted_value_starting_from_idx(rewards, idx):

    return np.sum(np.array(rewards[idx:]) * (DISCOUNT_FACTOR ** np.arange(len(rewards[idx:]))))


def calculate_approx_value_function(policy, n_episodes=500, learning_rate=0.000_001):

    # Return approximate value function (or weights for approx value function?) for agent following this policy
    # Follows Sutton and Barto 9.3 Gradient Monte Carlo for Estimating v_pi

    policy_interpolator = RegularGridInterpolator(points=(WEALTH_GRID, INCOME_GRID, AGE_GRID), values=policy, method="linear", bounds_error=False, fill_value=None)

    # The shape of the coefficient vector needs to be consistent with get_feature_vector
    value_fn_coefficients = np.ones((feature_vector_size(), ))

    dfs = []

    for episode_idx in range(n_episodes):        

        reached_terminal_state = False
        next_wealth, next_income, next_age = (INITIAL_WEALTH, INITIAL_INCOME, MIN_AGE)

        rewards, predicted_values, feature_vectors = ([], [], [])
        consumption_fractions = []

        while not reached_terminal_state:

            wealth, income, age = (next_wealth, next_income, next_age)

            # This tells us whether the _current_ state is terminal (as opposed to next period's state)
            reached_terminal_state = is_terminal_state(wealth, income, age)

            feature_vector = get_feature_vector(wealth, income, age)
            feature_vectors.append(feature_vector)
            
            predicted_value = np.dot(feature_vector, value_fn_coefficients)
            predicted_values.append(predicted_value)

            consumption_fraction = policy_interpolator(np.array([wealth, income, age]))[0]

            # Convention: the policy tells us the _fraction_ of wealth + income that the agent will consume in the current period
            consumption = consumption_fraction * (wealth + income)
            consumption_fractions.append(consumption_fraction)

            reward = reward_function(consumption)
            rewards.append(reward)

            discounted_predicted_continuation_value = 0
            if not reached_terminal_state:
                next_wealth, next_income, next_age = simulate_next_state(wealth, income, age, consumption)

        actual_values = []

        for idx, predicted_value in enumerate(predicted_values):
            feature_vector = feature_vectors[idx]
            actual_value = discounted_value_starting_from_idx(rewards, idx)
            actual_values.append(actual_value)
            value_fn_coefficients = value_fn_coefficients + learning_rate * (actual_value - predicted_value) * feature_vector

        # Messy, these colnames need to be consistent with get_feature_vector
        columns = [
            "constant",
            "log_wealth",
            "income",
            "age_scaled",
            "log_wealth_income",
            "log_wealth_age",
            "income_age",
        ]
        df = pd.DataFrame(feature_vectors, columns=columns)
        df["age"] = df["age_scaled"] * 100
        df["wealth"] = np.exp(df["log_wealth"])
        df["actual_value"] = actual_values
        df["reward"] = rewards
        df["consumption"] = consumption_fractions * (df["wealth"] + df["income"])
        df["consumption_fractions"] = consumption_fractions
        df["episode_idx"] = episode_idx
        dfs.append(df)
        # print(f"Done running episode {episode_idx}, coefficients {np.round(value_fn_coefficients, 2)}")

    df = pd.concat(dfs)

    return value_fn_coefficients, df


def update_policy(value_fn_coefficients, old_policy):

    # Policy is input is (wealth, income, age)
    # TODO Could try solving with first order condition instead of brute force, compare results

    policy = np.zeros_like(old_policy)

    for wealth_idx, wealth in enumerate(WEALTH_GRID):
        print(f"Updating policy, wealth idx {wealth_idx} of {WEALTH_GRID.size - 1}")
        for income_idx, income in enumerate(INCOME_GRID):
            for age_idx, age in enumerate(AGE_GRID):
                
                # Shortcut: optimal policy in final year is to consume everything
                if age >= MAX_AGE:
                    policy[wealth_idx, income_idx, age_idx] = 1.0
                    continue

                # Another shortcut: for states that aren't reachable (positive income after retirement), fill things in quickly
                if age >= RETIREMENT_AGE and income_idx > 0:
                    policy[wealth_idx, income_idx, age_idx] = policy[wealth_idx, 0, age_idx]
                    continue

                # Candidate values for consumption
                consumption = CONSUMPTION_FRACTION_GRID * (income + wealth)

                reward = reward_function(consumption)                
                
                next_wealth = (wealth + income - consumption) * (1 + INTEREST_RATE)
                next_age = (age + 1) * np.ones_like(next_wealth)

                if next_age[0] >= RETIREMENT_AGE:
                    next_income = np.zeros_like(next_wealth)
                    next_state = np.stack([next_wealth, next_income, next_age])
                    features = np.apply_along_axis(lambda x: get_feature_vector(*x), 0, next_state)
                    continuation_value = np.dot(value_fn_coefficients, features)
                        
                else:
                    # The next state is random (because income is random)
                    # We need to calculate an expected value
                    expected_next_income = RAISE_PROBABILITY * income * (1 + (MIN_RAISE + MAX_RAISE) / 2) + (1 - RAISE_PROBABILITY) * income
                    expected_next_income = expected_next_income * np.ones_like(next_wealth)
                    
                    expected_next_state = np.stack([next_wealth, expected_next_income, next_age])

                    features = np.apply_along_axis(lambda x: get_feature_vector(*x), 0, expected_next_state)
                    continuation_value = np.dot(value_fn_coefficients, features)

                value = reward + DISCOUNT_FACTOR * continuation_value

                best_consumption_fraction = CONSUMPTION_FRACTION_GRID[np.argmax(value)]

                policy[wealth_idx, income_idx, age_idx] = best_consumption_fraction

    print("Policy fn sanity check")
    correct_policy_penultimate_period = 1 / (1 + DISCOUNT_FACTOR)  # TODO Compare to policy[:, 0, AGE_GRID.size - 2]

    # TODO Another sanity check:  plot correct value function in final period v(w) = ln(w) versus approximate value fn

    # import pdb; pdb.set_trace()

    return policy

                
def main(n_policy_updates=5):

    # These are _fractions_ of available wealth+income to consume in the current period
    # State is wealth, income, age
    # Note that the grid includes states that are not reachable!  For example, the grid includes
    # states where the agent is above RETIREMENT_AGE but still earns income
    # TODO Could have a separate grid for before and after retirement age (fewer points to loop over)
    # The optimal policy after retiring is very similar to the cake eating problem!
    policy = np.min(CONSUMPTION_FRACTION_GRID) * np.ones((WEALTH_GRID.size, INCOME_GRID.size, AGE_GRID.size))    

    for policy_update_idx in range(n_policy_updates):

        value_fn_coefficients, df = calculate_approx_value_function(policy)
        policy = update_policy(value_fn_coefficients, policy)

        # TODO Run sanity checks on policy (check last and before last period, for example)

        # TODO Put all of these plots in separate functions
        fig, ax = plt.subplots(figsize=(10, 8))

        plt.scatter(df["age"], df["wealth"], alpha=0.1)

        plt.xlabel("age")
        plt.ylabel("wealth (at beginning of period)")

        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOT_DIR)

        outfile = f"simulations_age_and_wealth_iteration_{policy_update_idx}.png"
        plt.savefig(os.path.join(outdir, outfile))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))

        plt.scatter(df["age"], df["income"], alpha=0.1)

        plt.xlabel("age")
        plt.ylabel("income")

        outfile = f"simulations_age_and_income_iteration_{policy_update_idx}.png"
        plt.savefig(os.path.join(outdir, outfile))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))

        plt.scatter(df["age"], df["consumption"], alpha=0.1)

        plt.xlabel("age")
        plt.ylabel("consumption")

        outfile = f"simulations_age_and_consumption_iteration_{policy_update_idx}.png"
        plt.savefig(os.path.join(outdir, outfile))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))

        plt.scatter(df["age"], df["consumption_fractions"], alpha=0.1)

        plt.xlabel("age")
        plt.ylabel("consumption fraction")

        outfile = f"simulations_age_and_consumption_fraction_iteration_{policy_update_idx}.png"
        plt.savefig(os.path.join(outdir, outfile))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))

        for wealth_idx in [0, 5, 10]:
            for income_idx in [0, 5, 10]:
                label = f"wealth = {np.round(WEALTH_GRID[wealth_idx], 2)}, income = {np.round(INCOME_GRID[income_idx], 2)}"
                plt.plot(AGE_GRID, policy[wealth_idx, income_idx, :], label=label)

        plt.xlabel("age (part of the state variable)")
        plt.ylabel("fraction of wealth + income consumed in current period (action)")
        ax.legend()

        outfile = f"policy_fraction_consumed_as_of_iteration_{policy_update_idx}.png"
        plt.savefig(os.path.join(outdir, outfile))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))

        for wealth_idx in [0, 5, 10]:
            for income_idx in [0, 5, 10]:
                label = f"wealth = {np.round(WEALTH_GRID[wealth_idx], 2)}, income = {np.round(INCOME_GRID[income_idx], 2)}"
                wealth = WEALTH_GRID[wealth_idx]
                income = INCOME_GRID[income_idx]
                amount_consumed = (wealth + income) * policy[wealth_idx, income_idx, :]
                plt.plot(AGE_GRID, amount_consumed, label=label)

        plt.xlabel("age (part of the state variable)")
        plt.ylabel("amount consumed in current period (action)")
        ax.legend()

        outfile = f"policy_amount_consumed_as_of_iteration_{policy_update_idx}.png"
        plt.savefig(os.path.join(outdir, outfile))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        for age_idx in [0, 5, 10]:
            for income_idx in [0, 5]:
                age = AGE_GRID[age_idx]
                income = INCOME_GRID[income_idx]
                state = np.stack([WEALTH_GRID, income * np.ones_like(WEALTH_GRID), age * np.ones_like(WEALTH_GRID)])
                features = np.apply_along_axis(lambda x: get_feature_vector(*x), 0, state)
                approx_value = np.dot(value_fn_coefficients, features)

                label = f"age = {age}, income = {np.round(income, 2)}"
                plt.plot(WEALTH_GRID, approx_value, label=label)

        plt.xlabel("wealth (part of the state variable)")
        plt.ylabel("approx value fn")
        ax.legend()
        outfile = f"approx_value_as_of_iteration_{policy_update_idx}.png"
        plt.savefig(os.path.join(outdir, outfile))
        plt.close()


if __name__ == "__main__":
    main()
