import numpy as np

# Note: actions are tuples of change in (x, y) position
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

# Note: earn this reward in every period (undiscounted) until objective is reached
# Goal is to reach the objective (target location) as fast as possible
REWARD = -1


def is_valid_probability(x):
    return 0.0 <= x <= 1.0, "Probabilities must be between 0 and 1 (inclusive)"


def is_valid_discount_factor(x):
    return 0.0 < x < 1.0, "Discount factor must be between 0 and 1 (exclusive)"


class WindyGridworld:
    def __init__(
        self,
        width=8,
        height=10,
        pr_wind_up=0.1,
        pr_wind_down=0.0,
        windy_columns=[2, 3, 4],
        discount=0.90,
    ):

        assert is_valid_probability(pr_wind_down)
        assert is_valid_probability(pr_wind_up)

        pr_wind_stay = 1.0 - pr_wind_down - pr_wind_up

        self.pr_wind_up = pr_wind_up
        self.pr_wind_stay = pr_wind_down
        self.pr_wind_down = pr_wind_down

        assert all([0 <= column_index < width for column_index in windy_columns])

        self.windy_columns = windy_columns

        assert is_valid_discount_factor(discount)

        self.discount = discount

        self.value = np.zeros((width, height))

        # Note: policy[:, :, 0] is movement in x dimension, and policy[:, :, 1] is movement in y dimension
        self.policy = np.zeros((width, height, 2), dtype=int)

    def is_target_location(self, x, y):

        width, height = self.value.shape

        # Note: target location is top right corner of grid (if origin is bottom left)
        return x >= width - 1 and y >= height - 1

    def get_x_y_next(self, x, y, action):

        x_next, y_next = (x + action[0], y + action[1])

        width, height = self.value.shape

        # Note: this prevents the next (x, y) location from being outside of the grid
        x_next = min(max(x_next, 0), width - 1)
        y_next = min(max(y_next, 0), height - 1)

        return x_next, y_next

    def get_updated_value_at_location(self, x, y, action):

        if self.is_target_location(x, y):

            # Note: value is zero when you reach the target
            return 0.0

        x_next, y_next = self.get_x_y_next(x, y, action)

        # TODO Expected continuation value, wind
        continuation_value = self.value[x_next, y_next]

        return REWARD + self.discount * continuation_value

    def get_updated_value_function(self):

        updated_value = self.value.copy()

        width, height = self.value.shape

        for x in range(width):
            for y in range(height):

                action = tuple(self.policy[x, y])
                assert (
                    action in ACTIONS
                ), "Uh oh! There's an invalid action in the policy function"

                # Note: this uses self.value, i.e. the value function from the previous iteration,
                # in addition to the policy function
                updated_value[x, y] = self.get_updated_value_at_location(x, y, action)

        return updated_value

    def solve_value_fn(self, max_iterations=50, value_epsilon=0.001):

        for iteration in range(max_iterations):

            updated_value = self.get_updated_value_function()

            self.value = updated_value

    def get_updated_policy_function(self):

        updated_policy = self.policy.copy()

        width, height = self.value.shape

        for x in range(width):
            for y in range(height):

                candidate_values = [
                    self.get_updated_value_at_location(x, y, action)
                    for action in ACTIONS
                ]

                optimal_action = ACTIONS[np.argmax(candidate_values)]

                updated_policy[x, y] = optimal_action

        return updated_policy

    def run_policy_iteration(self, max_iterations=50, policy_epsilon=0.1):

        for iteration in range(max_iterations):

            # Note: solve for the value function given the previous policy
            self.solve_value_fn()

            # Note: given the updated value function, find the optimal policy
            updated_policy = self.get_updated_policy_function()

            self.policy = updated_policy


def main():

    gridworld = WindyGridworld()

    gridworld.run_policy_iteration()

    print(gridworld.policy)
    print(gridworld.value)


if __name__ == "__main__":
    main()
