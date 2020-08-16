import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Note: actions are tuples of change in (x, y) position, i.e. (dx, dy)
#  Valid actions are stay, left, right, down, up
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

# Note: the agent earns this reward in every period until
#  the objective is reached (which ends the episode)
#  The goal is to reach the objective (target location) as fast as possible
REWARD_DEFAULT = -1

# Note: certain locations are "obstacles" which are passable but very costly
#  Since REWARD_OBSTACLE < REWARD_DEFAULT, the agent would like to avoid these locations
REWARD_OBSTACLE = -10

# Note: the problem ends when the agent reaches the target
REWARD_TARGET = 0

# Note: these are algorithm names
POLICY_ITERATION = "policy_iteration"
SARSA = "sarsa"
Q_LEARNING = "q_learning"

PLOT_FILENAME = "value_and_policy_functions_solved_by_{algorithm}_{wind_description}"
PLOT_DIR = "plots"

PR_WIND_UP = 0.3
PR_WIND_DOWN = 0.3


def is_valid_probability(x):
    return 0.0 <= x <= 1.0, "Probabilities must be between 0 and 1 (inclusive)"


def is_valid_discount_factor(x):
    return 0.0 <= x <= 1.0, "Discount factor must be between 0 and 1 (inclusive)"


class WindyGridworld:
    def __init__(
        self,
        pr_wind_up,
        pr_wind_down,
        width=10,
        height=12,
        target_xy=[6, 8],
        obstacles_xy=[[5, 8], [6, 6]],
        discount=1.0,
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
        #  with algorithm names as their keys (for example, self.value[POLICY_ITERATION]
        #  will store the value function estimated by policy iteration)
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

    def get_xy_next(self, x, y, dx_and_dy):

        # Note: the y component of dx_and_dy is the sum of the agent's action and stochastic wind

        x_next, y_next = (x + dx_and_dy[0], y + dx_and_dy[1])

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

        reward = self.get_reward(x, y)

        if self.is_target_location(x, y):

            # Note: there is no continuation value when the agent reaches the target
            #  (they receive REWARD_TARGET and the episode ends)
            return reward

        expected_continuation_value = 0.0

        for probability, dy_from_wind in [
            (self.pr_wind_up, 1),
            (self.pr_wind_stay, 0),
            (self.pr_wind_down, -1),
        ]:
            # Note: wind randomly modifies action's vertical (y-axis) component
            dx_and_dy = (action[0], action[1] + dy_from_wind)
            x_next, y_next = self.get_xy_next(x, y, dx_and_dy)

            expected_continuation_value += (
                probability * self.value[POLICY_ITERATION][x_next, y_next]
            )

        return reward + self.discount * expected_continuation_value

    def get_updated_value_function(self):

        updated_value = self.value[POLICY_ITERATION].copy()

        for x in range(self.width):
            for y in range(self.height):

                action = tuple(self.policy[POLICY_ITERATION][x, y])
                assert (
                    action in ACTIONS
                ), "Uh oh! There's an invalid action in the policy function"

                # Note: this uses self.value, i.e. the value function from the previous iteration,
                #  in addition to the policy function
                updated_value[x, y] = self.get_updated_value_at_location(x, y, action)

        return updated_value

    def solve_value_fn(self, max_iterations=50, value_epsilon=0.00001):

        # TODO Could be fun to also solve the value fn by sparse matrix inversion,
        #  which is probably feasible given the problem size

        for iteration in range(max_iterations):

            updated_value = self.get_updated_value_function()

            distance = np.max(np.abs(updated_value - self.value[POLICY_ITERATION]))

            if distance < value_epsilon:
                # Note: break early if the value function has already converged
                break

            self.value[POLICY_ITERATION] = updated_value

    def get_locations_with_ties_in_policy_function(self):

        locations_with_ties = []
        alternate_optimal_actions = []

        for x in range(self.width):
            for y in range(self.height):

                # Note: technically, all actions are optimal at the target location
                #  (because the episode ends), but the plots look nicer without alternate
                #  optimal actions at the target location
                if self.is_target_location(x, y):
                    continue

                candidate_values = [
                    self.get_updated_value_at_location(x, y, action)
                    for action in ACTIONS
                ]

                optimal_value = np.max(candidate_values)

                if np.sum(np.isclose(candidate_values, optimal_value)) > 1:

                    locations_with_ties.append((x, y))

                    optimal_action_indices = np.where(
                        np.isclose(candidate_values, optimal_value)
                    )[0]
                    alternate_optimal_actions.append(
                        [ACTIONS[index] for index in optimal_action_indices[1:]]
                    )

        return locations_with_ties, alternate_optimal_actions

    def get_updated_policy_function(self):

        updated_policy = self.policy[POLICY_ITERATION].copy()

        for x in range(self.width):
            for y in range(self.height):

                candidate_values = [
                    self.get_updated_value_at_location(x, y, action)
                    for action in ACTIONS
                ]

                # Note: the argmax docs state that "In case of multiple occurrences of the maximum values,
                #  the indices corresponding to the first occurrence are returned."
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

        (
            self.locations_with_ties,
            self.alternate_optimal_actions,
        ) = self.get_locations_with_ties_in_policy_function()

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

        dx_and_dy = (action[0], action[1] + dy_from_wind)
        x_next, y_next = self.get_xy_next(x, y, dx_and_dy)

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
                    reward
                    + self.discount * Q_next
                    - self.Q[Q_LEARNING][x, y, action_idx]
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

    def get_simulated_path(self, initial_xy, max_iterations=1000):

        x, y = initial_xy

        xs = [x]
        ys = [y]

        for time in range(max_iterations):

            action = tuple(self.policy[POLICY_ITERATION][x, y])
            x, y = self.simulate_xy_next(x, y, action)

            xs.append(x)
            ys.append(y)

            if self.is_target_location(x, y):
                return xs, ys

        return xs, ys

    def save_plots_of_simulated_paths(self, n_simulations=1000, alpha=0.05):

        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOT_DIR)

        for initial_xy in [(0, 0), (0, 7), (7, 1), (9, 10)]:

            fig, ax = plt.subplots()

            # Note: imshow puts first index along vertical axis,
            # so we swap axes / transpose to put y along the vertical axis and x along the horizontal
            im = ax.imshow(np.transpose(self.value[POLICY_ITERATION]), origin="lower")

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("value V(s)", rotation=-90, va="bottom")

            plt.xlabel("x")
            plt.ylabel("y")

            for _ in range(n_simulations):

                xs, ys = self.get_simulated_path(initial_xy)
                plt.plot(xs, ys, color="black", alpha=alpha)

            plt.title(f"Simulated Paths Starting From {initial_xy}")

            outfile = (
                f"simulated_paths_starting_from_{initial_xy[0]}_{initial_xy[1]}.png"
            )
            plt.savefig(os.path.join(outdir, outfile))
            plt.clf()

    def get_wind_description(self):

        if np.allclose([self.pr_wind_up, self.pr_wind_down], 0.0):
            return "Without Wind"

        return "With Wind"

    def save_value_and_policy_function_plot(self, algorithm):

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

        if algorithm == POLICY_ITERATION:

            for location_with_ties, alternate_optimal_actions in zip(
                self.locations_with_ties, self.alternate_optimal_actions
            ):

                actions_x, actions_y = list(zip(*alternate_optimal_actions))
                ax.quiver(
                    location_with_ties[0], location_with_ties[1], actions_x, actions_y
                )

        plt.xlabel("x")
        plt.ylabel("y")

        wind_description = self.get_wind_description()
        plt.title(f"Value and Policy Functions {wind_description}")

        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOT_DIR)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        wind_description_for_filename = wind_description.lower().replace(" ", "_")

        outfile = PLOT_FILENAME.format(
            algorithm=algorithm, wind_description=wind_description_for_filename
        )
        plt.savefig(os.path.join(outdir, outfile))


def main():

    gridworld = WindyGridworld(pr_wind_up=PR_WIND_UP, pr_wind_down=PR_WIND_DOWN)

    gridworld.run_policy_iteration()
    gridworld.save_value_and_policy_function_plot(POLICY_ITERATION)
    gridworld.save_plots_of_simulated_paths()

    gridworld.run_q_learning()
    gridworld.save_value_and_policy_function_plot(Q_LEARNING)

    gridworld.run_sarsa()
    gridworld.save_value_and_policy_function_plot(SARSA)

    windless_gridworld = WindyGridworld(pr_wind_up=0.0, pr_wind_down=0.0)

    windless_gridworld.run_policy_iteration()
    windless_gridworld.save_value_and_policy_function_plot(POLICY_ITERATION)

    # Notice that the value function is generally (but not always!) larger in the windless case
    print("Differences in value function (value without wind - value with wind)")
    diff_in_value = (
        windless_gridworld.value[POLICY_ITERATION] - gridworld.value[POLICY_ITERATION]
    )
    print(diff_in_value.round(1))


if __name__ == "__main__":
    main()
