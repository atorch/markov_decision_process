import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


DISCOUNT_FACTOR = 0.9
VALUE_FUNCTION_CONSTANT_TERM = (
    np.log(1 - DISCOUNT_FACTOR)
    + np.log(DISCOUNT_FACTOR) * DISCOUNT_FACTOR / (1 - DISCOUNT_FACTOR)
) / (1 - DISCOUNT_FACTOR)

PLOT_DIR = "plots"

# Note: these grids are used for naive searches over the action space
#  Where action = fraction_consumed * wealth
FRACTION_CONSUMED_GRID_COARSE = np.linspace(0.01, 0.99, 4)
FRACTION_CONSUMED_GRID_FINE = np.linspace(0.01, 0.99, 8)


def reward_function(action):

    # Note: the agent's action is the amount they consume in the current period
    #  If they consume nothing, they receive a reward of negative infinity (they die!)
    #  If they consume everything, they receive a large reward in the current period, but they die tomorrow
    #  The optimal action must therefore be to consume something (but not everything)
    return np.log(action)


def optimal_policy(state):

    # Note: the agent's state is their wealth (their "amount of cake")
    #  They need to decide how much to consume today and how much to leave for future periods
    #  The agent lives forever and is unemployed: their wealth can never increase; it can only
    #  decrease depending on how much they consume. The optimal policy can be solved with
    #  pen and paper. A more patient agent (with a higher discount factor) consumes less today
    #  and saves more for future periods
    return (1 - DISCOUNT_FACTOR) * state


def optimal_value_function(state):

    # Note: this is the value of pursuing the optimal policy
    return (1 / (1 - DISCOUNT_FACTOR)) * np.log(state) + VALUE_FUNCTION_CONSTANT_TERM


def get_next_state(state, action):

    # Note: the agent's action is how much to consume today.
    #  Whatever is not consumed today is available tomorrow.
    #  Wealth is a stock, consumption is a flow.
    return state - action


def optimal_policy_grid_search(
    state, approximate_value_function, fraction_consumed_grid
):

    # Note: the grid is in [0, 1], and the action is equal to wealth * fraction consumed
    state_mesh, fraction_consumed_mesh = np.meshgrid(state, fraction_consumed_grid)

    actions = state_mesh * fraction_consumed_mesh
    rewards = reward_function(actions)

    next_states = get_next_state(state_mesh, actions)

    log_next_states = np.log(next_states.reshape(-1, 1))
    continuation_values = approximate_value_function.predict(log_next_states).reshape(
        actions.shape
    )

    candidate_values = rewards + DISCOUNT_FACTOR * continuation_values

    argmax_candidate_values = np.argmax(candidate_values, axis=0)

    return actions[argmax_candidate_values, range(state.size)]


def optimal_policy_given_approximate_value_function(state, approximate_value_function):

    log_wealth_coefficient = approximate_value_function.coef_[0]

    # Note: on the first iteration, the coefficient on log wealth is zero (future wealth has no value),
    #  so, without this shortcut, the agent would consume everything immediately and get a -Inf continuation value
    if log_wealth_coefficient <= 0.0:
        return state * 0.99

    # Note: to arrive at this policy, write down the Bellman equation using the
    #  approximate value function as the continuation value, and optimize with respect to the action
    return state / (DISCOUNT_FACTOR * log_wealth_coefficient + 1)


def get_estimated_values(
    states, approximate_value_function, get_optimal_policy, **kwargs
):

    actions = get_optimal_policy(states, approximate_value_function, **kwargs)

    rewards = reward_function(actions)
    next_states = get_next_state(states, actions)

    # Note: the approximated value function takes log(state) as input and returns an estimated value
    log_next_states = np.log(next_states.reshape(-1, 1))

    continuation_values = approximate_value_function.predict(log_next_states)

    # Note: this is the Bellman equation
    return rewards + DISCOUNT_FACTOR * continuation_values


def get_coefficients(linear_regression):

    return np.vstack([linear_regression.intercept_, linear_regression.coef_])


def calculate_approximate_solution(
    get_optimal_policy, max_iterations=10000, n_simulations=2000, **kwargs
):

    X = np.zeros((n_simulations, 1))
    y = np.zeros((n_simulations,))
    approximate_value_function = LinearRegression()
    approximate_value_function.fit(X=X, y=y)

    print(f"running solver using {get_optimal_policy} to find actions given estimated value function")

    for i in range(max_iterations):

        states = np.random.uniform(low=0.001, high=5.0, size=n_simulations)
        X[:, 0] = np.log(states)
        estimated_values = get_estimated_values(
            states, approximate_value_function, get_optimal_policy, **kwargs
        )
        y = estimated_values

        previous_coefficients = get_coefficients(approximate_value_function)

        approximate_value_function.fit(X=X, y=y)
        current_coefficients = get_coefficients(approximate_value_function)

        if np.allclose(
            current_coefficients, previous_coefficients, rtol=1e-04, atol=1e-06
        ):
            print(f"converged at iteration {i}!")
            break

    print(
        f"true value is {(1 / (1 - DISCOUNT_FACTOR))}, estimate is {approximate_value_function.coef_}"
    )
    print(
        f"true value is {VALUE_FUNCTION_CONSTANT_TERM}, estimate is {approximate_value_function.intercept_}"
    )

    return approximate_value_function


def save_value_function_plot(
    approximate_value_function,
    approximate_value_function_coarse_grid,
    approximate_value_function_fine_grid,
):

    fig, ax = plt.subplots(figsize=(10, 8))

    # Note: wealth is the state variable
    wealth = np.linspace(0.01, 100, 1000)
    log_wealth = np.log(wealth).reshape(-1, 1)

    correct_value = optimal_value_function(wealth)
    approximate_value = approximate_value_function.predict(log_wealth)
    approximate_value_coarse_grid = approximate_value_function_coarse_grid.predict(
        log_wealth
    )
    approximate_value_fine_grid = approximate_value_function_fine_grid.predict(
        log_wealth
    )

    plt.plot(wealth, correct_value, label="true value function (analytical solution)")

    # Note: don't show the left and right ends of the estimated value functions
    #  so that they don't entirely cover/hide the true value function on the plot
    plt.plot(
        wealth[1:-3],
        approximate_value[1:-3],
        "--",
        label="estimated value function (using log-linear regression & first order condition for action)",
    )
    plt.plot(
        wealth[1:-3],
        approximate_value_coarse_grid[1:-3],
        ":",
        label="estimated value function (using log-linear regression & coarse grid search for action)",
    )
    plt.plot(
        wealth[1:-3],
        approximate_value_fine_grid[1:-3],
        ":",
        label="estimated value function (using log-linear regression & fine grid search for action)",
    )

    plt.xlabel("wealth (state variable)")
    plt.ylabel("value function")
    ax.legend()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOT_DIR)
    outfile = "cake_eating_problem_value_function.png"
    plt.savefig(os.path.join(outdir, outfile))


def main():

    approximate_value_function = calculate_approximate_solution(
        optimal_policy_given_approximate_value_function
    )
    approximate_value_function_coarse_grid = calculate_approximate_solution(
        optimal_policy_grid_search, fraction_consumed_grid=FRACTION_CONSUMED_GRID_COARSE
    )
    approximate_value_function_fine_grid = calculate_approximate_solution(
        optimal_policy_grid_search, fraction_consumed_grid=FRACTION_CONSUMED_GRID_FINE
    )

    save_value_function_plot(
        approximate_value_function,
        approximate_value_function_coarse_grid,
        approximate_value_function_fine_grid,
    )


if __name__ == "__main__":
    main()
