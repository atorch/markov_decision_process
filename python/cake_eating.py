import numpy as np
from sklearn.linear_model import LinearRegression


DISCOUNT_FACTOR = 0.9


def reward_function(action):

    # Note: the agent's action is the amount they consume in the current period
    #  If they consume nothing, they receive a reward of negative infinity (they die!)
    #  If they consume everything, they receive a large reward in the current period, but they die tomorrow
    return np.log(action)


def optimal_policy(state):

    # Note: the agent's state is their wealth (their "amount of cake")
    #  They need to decide how much to consume today and how much to leave for future periods
    #  The agent lives forever and is unemployed: their wealth can never increase; it can only
    #  decrease depending on how much they consume. The optimal policy can be solved with
    #  pen and paper
    return (1 - DISCOUNT_FACTOR) * state


def optimal_value_function(state):

    # TODO This isn't correct, there's also a constant term
    return (1 / (1 - DISCOUNT_FACTOR)) * np.log(state)


def get_next_state(state, action):

    # Note: the agent's action is how much to consume today.
    #  Whatever is not consumed today is available tomorrow.
    #  Wealth is a stock, consumption is a flow.
    return state - action


def get_estimated_values(states, approximate_value_function):

    # Hack. TODO: replace optimal policy with a search for optimal policy
    actions = optimal_policy(states)
    rewards = reward_function(actions)
    next_states = get_next_state(states, actions)

    # Note: the approximated value function takes log(state) as input and returns an estimated value
    log_next_states = np.log(next_states.reshape(-1, 1))

    continuation_values = approximate_value_function.predict(log_next_states)

    return rewards + DISCOUNT_FACTOR * continuation_values


def calculate_approximate_solution(max_iterations=10000, n_simulations=2000):

    X = np.zeros((n_simulations, 1))
    y = np.zeros((n_simulations, ))
    approximate_value_function = LinearRegression()
    approximate_value_function.fit(X=X, y=y)

    for i in range(max_iterations):

        states = np.random.uniform(low=0.001, high=5.0, size=n_simulations)
        X[:, 0] = np.log(states)
        estimated_values = get_estimated_values(states, approximate_value_function)
        y = estimated_values

        previous_coefficients = np.vstack([approximate_value_function.intercept_, approximate_value_function.coef_])

        approximate_value_function.fit(X=X, y=y)
        current_coefficients = np.vstack([approximate_value_function.intercept_, approximate_value_function.coef_])

        if np.allclose(current_coefficients, previous_coefficients, rtol=1e-03, atol=1e-05):
            print(f"converged at iteration {i}!")
            break

    print(f"true value is {(1 / (1 - DISCOUNT_FACTOR))}, estimate is {approximate_value_function.coef_}")


def main():

    calculate_approximate_solution()

if __name__ == "__main__":
    main()
