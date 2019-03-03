import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Note: actions are tuples of change in (x, y) position, i.e. (dx, dy)
# Valid actions are stay, left, right, down, up
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

# Note: earn this reward in every period until objective is reached (which ends the episode)
# With a negative reward, the goal is to reach the objective (target location) as fast as possible
REWARD_DEFAULT = -1

# Note: certain locations can be "obstacles" which are passable but very costly
REWARD_OBSTACLE = -10


def is_valid_probability(x):
    return 0.0 <= x <= 1.0, "Probabilities must be between 0 and 1 (inclusive)"


def is_valid_discount_factor(x):
    return 0.0 < x < 1.0, "Discount factor must be between 0 and 1 (exclusive)"


class WindyGridworld:
    def __init__(
        self,
        width=10,
        height=12,
        target_xy=[6, 8],
        obstacles_xy=[[5, 8], [6, 6]],
        pr_wind_up=0.1,
        pr_wind_down=0.2,
        discount=0.90,
    ):

        assert is_valid_probability(pr_wind_down)
        assert is_valid_probability(pr_wind_up)

        pr_wind_stay = 1.0 - pr_wind_down - pr_wind_up

        self.pr_wind_up = pr_wind_up
        self.pr_wind_stay = pr_wind_stay
        self.pr_wind_down = pr_wind_down

        assert np.isclose(self.pr_wind_up + self.pr_wind_stay + self.pr_wind_down, 1.0)

        # Note: make sure target location is inside grid
        assert 0 <= target_xy[0] < width
        assert 0 <= target_xy[1] < height

        self.target_x, self.target_y = target_xy

        for obstacle_xy in obstacles_xy:
            # Note: make sure each obstacle location is inside grid
            assert 0 <= obstacle_xy[0] < width
            assert 0 <= obstacle_xy[1] < height

        self.obstacles_xy = obstacles_xy

        assert is_valid_discount_factor(discount)

        self.discount = discount

        self.value = np.zeros((width, height))

        # Note: policy[:, :, 0] is movement in x dimension, and policy[:, :, 1] is movement in y dimension
        self.policy = np.zeros((width, height, 2), dtype=int)

    def is_target_location(self, x, y):

        return x == self.target_x and y == self.target_y

    def is_obstacle_location(self, x, y):

        return any(
            [
                x == obstacle_xy[0] and y == obstacle_xy[1]
                for obstacle_xy in self.obstacles_xy
            ]
        )

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

        continuation_value = 0.0

        for probability, dy_from_wind in [
            (self.pr_wind_up, 1),
            (self.pr_wind_stay, 0),
            (self.pr_wind_down, -1),
        ]:

            # Note: wind randomly modifies action's vertical (y) movement
            action_plus_wind = (action[0], action[1] + dy_from_wind)
            x_next, y_next = self.get_x_y_next(x, y, action_plus_wind)

            continuation_value += probability * self.value[x_next, y_next]

        reward = REWARD_DEFAULT

        if self.is_obstacle_location(x, y):
            reward = REWARD_OBSTACLE

        return reward + self.discount * continuation_value

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

    def run_policy_iteration(self, max_iterations=50, verbose=True):

        for iteration in range(max_iterations):

            # Note: solve for the value function given the previous policy
            self.solve_value_fn()

            if verbose:
                print(self.value.round(1))

            # Note: given the updated value function, find the optimal policy
            updated_policy = self.get_updated_policy_function()

            if np.all(self.policy == updated_policy):
                print(f"Policy function converged after {iteration} iteration(s)")
                break

            self.policy = updated_policy

    def save_value_and_policy_function_plot(self, outfile):
        fig, ax = plt.subplots()

        # Note: imshow puts first index along vertical axis,
        # so we swap axes / transpose to put y along the vertical axis and x along the horizontal
        im = ax.imshow(np.transpose(self.value), origin="lower")

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("value V(s)", rotation=-90, va="bottom")

        width, height = self.value.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")

        ax.quiver(
            x.flatten(),
            y.flatten(),
            self.policy[:, :, 0].flatten(),
            self.policy[:, :, 1].flatten(),
        )

        plt.plot(self.target_x, self.target_y, color="black", marker="x")

        for obstacle_xy in self.obstacles_xy:

            plt.plot(
                obstacle_xy[0],
                obstacle_xy[1],
                color="firebrick",
                fillstyle="none",
                marker="o",
            )

        plt.xlabel("x")
        plt.ylabel("y")

        plt.title("Value and Policy Functions")

        plt.savefig(outfile)


def main():

    gridworld = WindyGridworld()

    gridworld.run_policy_iteration()

    gridworld.save_value_and_policy_function_plot("value_and_policy_functions.png")


if __name__ == "__main__":
    main()
