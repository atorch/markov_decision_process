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

# Note: the problem ends when the agent reaches the target
REWARD_TARGET = 0

# Note: these are algorithm names (dictionary keys)
POLICY_ITERATION = "policy_iteration"
SARSA = "sarsa"
Q_LEARNING = "q_learning"

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

        assert is_valid_probability(pr_wind_stay)

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

        self.width = width
        self.height = height

        # Note: value, policy, and Q functions will be saved in dictionaries
        # with algorithm names as their keys (for example, self.value[POLICY_ITERATION]
        # will store the value function estimated by policy iteration)
        self.value, self.policy, self.Q = ({}, {}, {})

    def is_target_location(self, x, y):

        return x == self.target_x and y == self.target_y

    def is_obstacle_location(self, x, y):

        return any(
            [
                x == obstacle_xy[0] and y == obstacle_xy[1]
                for obstacle_xy in self.obstacles_xy
            ]
        )

    def get_xy_next(self, x, y, action_plus_wind):

        x_next, y_next = (x + action_plus_wind[0], y + action_plus_wind[1])

        # Note: this prevents the next (x, y) location from being outside of the grid
        x_next = min(max(x_next, 0), self.width - 1)
        y_next = min(max(y_next, 0), self.height - 1)

        return x_next, y_next

    def get_reward(self, x, y):

        if self.is_target_location(x, y):
            return REWARD_TARGET

        if self.is_obstacle_location(x, y):
            return REWARD_OBSTACLE

        return REWARD_DEFAULT

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
            # Note: wind randomly modifies action's vertical (y-axis) movement
            action_plus_wind = (action[0], action[1] + dy_from_wind)
            x_next, y_next = self.get_xy_next(x, y, action_plus_wind)

            continuation_value += (
                probability * self.value[POLICY_ITERATION][x_next, y_next]
            )

        reward = self.get_reward(x, y)

        return reward + self.discount * continuation_value

    def get_updated_value_function(self):

        updated_value = self.value[POLICY_ITERATION].copy()

        for x in range(self.width):
            for y in range(self.height):

                action = tuple(self.policy[POLICY_ITERATION][x, y])
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

            distance = np.max(np.abs(updated_value - self.value[POLICY_ITERATION]))

            if distance < value_epsilon:
                # Note: break early if the value function has already converged
                break

            self.value[POLICY_ITERATION] = updated_value

    def get_updated_policy_function(self):

        updated_policy = self.policy[POLICY_ITERATION].copy()

        for x in range(self.width):
            for y in range(self.height):

                candidate_values = [
                    self.get_updated_value_at_location(x, y, action)
                    for action in ACTIONS
                ]

                optimal_action = ACTIONS[np.argmax(candidate_values)]

                updated_policy[x, y] = optimal_action

        return updated_policy

    def run_policy_iteration(self, max_iterations=50, verbose=True):

        self.value[POLICY_ITERATION] = np.zeros((self.width, self.height))

        # Note: policy[:, :, 0] is movement in x dimension, and policy[:, :, 1] is movement in y dimension
        self.policy[POLICY_ITERATION] = np.zeros(
            (self.width, self.height, 2), dtype=int
        )

        for iteration in range(max_iterations):

            # Note: solve for the value function given the previous policy
            self.solve_value_fn()

            if verbose:
                print(self.value[POLICY_ITERATION].round(1))

            # Note: given the updated value function, find the optimal policy
            updated_policy = self.get_updated_policy_function()

            # Note: PDF of Sutton and Barto's Reinforcement Learning warns that
            # "this may never terminate if the policy continually switches
            #  between two or more policies that are equally good",
            # in which case this function will run until max_iterations
            # TODO Why was that note removed from the published text?
            if np.all(self.policy[POLICY_ITERATION] == updated_policy):
                print(f"Policy function converged after {iteration} iteration(s)")
                break

            self.policy[POLICY_ITERATION] = updated_policy

    def get_random_xy(self):

        x = np.random.choice(range(self.width))
        y = np.random.choice(range(self.height))

        return x, y

    def get_epsilon_greedy_action(self, x, y, epsilon, algorithm):

        random_uniform = np.random.uniform()

        if random_uniform < epsilon:
            action_idx = np.random.choice(range(len(ACTIONS)))

        else:
            action_idx = np.argmax(self.Q[algorithm][x, y])

        return ACTIONS[action_idx], action_idx

    def simulate_xy_next(self, x, y, action):

        dy_from_wind = np.random.choice(
            [1, 0, -1], p=[self.pr_wind_up, self.pr_wind_stay, self.pr_wind_down]
        )

        action_plus_wind = (action[0], action[1] + dy_from_wind)
        x_next, y_next = self.get_xy_next(x, y, action_plus_wind)

        return x_next, y_next

    def get_policy_from_Q(self, algorithm):
        policy = np.zeros((self.width, self.height, 2), dtype=int)

        for x in range(self.width):
            for y in range(self.height):

                policy[x, y] = ACTIONS[np.argmax(self.Q[algorithm][x, y])]

        return policy

    def run_q_learning(self, step_size=0.1, n_episodes=10000, epsilon=0.01):

        # Note: Q[x, y, idx] is the value of location (x, y) conditional on taking action ACTIONS[idx]
        self.Q[Q_LEARNING] = np.zeros((self.width, self.height, len(ACTIONS)))

        for episode in range(n_episodes):

            if episode % 1000 == 0:
                print(f"Q-learning episode {episode}")

            x, y = self.get_random_xy()

            while not self.is_target_location(x, y):

                # Note: this uses self.Q
                action, action_idx = self.get_epsilon_greedy_action(
                    x, y, epsilon, Q_LEARNING
                )

                x_next, y_next = self.simulate_xy_next(x, y, action)

                Q_next = np.max(self.Q[Q_LEARNING][x_next, y_next])

                reward = self.get_reward(x, y)

                self.Q[Q_LEARNING][x, y, action_idx] += step_size * (
                    reward + self.discount * Q_next - self.Q[Q_LEARNING][x, y, action_idx]
                )

                x, y = (x_next, y_next)

        # TODO Is np.max(self.Q, axis=2) a biased estimate of the value function?
        self.value[Q_LEARNING] = np.max(self.Q[Q_LEARNING], axis=2)

        self.policy[Q_LEARNING] = self.get_policy_from_Q(Q_LEARNING)

    def run_sarsa(self, step_size=0.1, n_episodes=10000):

        # Note: Q[x, y, idx] is the value of location (x, y) conditional on taking action ACTIONS[idx]
        self.Q[SARSA] = np.zeros((self.width, self.height, len(ACTIONS)))

        for episode in range(n_episodes):

            if episode % 1000 == 0:
                print(f"SARSA episode {episode}")

            # Note: this makes get_epsilon_greedy_action return the greedy (optimal) action in the limit
            epsilon = 1.0 / (episode + 1)

            x, y = self.get_random_xy()

            # Note: this uses self.Q
            action, action_idx = self.get_epsilon_greedy_action(x, y, epsilon, SARSA)

            while not self.is_target_location(x, y):

                x_next, y_next = self.simulate_xy_next(x, y, action)
                action_next, action_idx_next = self.get_epsilon_greedy_action(
                    x_next, y_next, epsilon, SARSA
                )

                Q_next = self.Q[SARSA][x_next, y_next, action_idx_next]

                reward = self.get_reward(x, y)

                self.Q[SARSA][x, y, action_idx] += step_size * (
                    reward + self.discount * Q_next - self.Q[SARSA][x, y, action_idx]
                )

                x, y = (x_next, y_next)
                action, action_idx = (action_next, action_idx_next)

        self.value[SARSA] = np.max(self.Q[SARSA], axis=2)

        self.policy[SARSA] = self.get_policy_from_Q(SARSA)

    def save_value_and_policy_function_plot(self, algorithm, outfile):

        fig, ax = plt.subplots()

        # Note: imshow puts first index along vertical axis,
        # so we swap axes / transpose to put y along the vertical axis and x along the horizontal
        im = ax.imshow(np.transpose(self.value[algorithm]), origin="lower")

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("value V(s)", rotation=-90, va="bottom")

        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing="ij")

        ax.quiver(
            x.flatten(),
            y.flatten(),
            self.policy[algorithm][:, :, 0].flatten(),
            self.policy[algorithm][:, :, 1].flatten(),
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
    gridworld.save_value_and_policy_function_plot(
        POLICY_ITERATION, "value_and_policy_functions.png"
    )

    gridworld.run_q_learning()
    gridworld.save_value_and_policy_function_plot(
        Q_LEARNING, "value_and_policy_functions_q_learning.png"
    )

    gridworld.run_sarsa()
    gridworld.save_value_and_policy_function_plot(
        SARSA, "value_and_policy_functions_sarsa.png"
    )


if __name__ == "__main__":
    main()
