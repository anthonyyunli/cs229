from __future__ import division, print_function
from env import CartPole, Physics
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
import os
from pathlib import Path

# Cart/pole dynamics and state discretization are inspired by the UMass RL repository:
# http://www-anw.cs.umass.edu/rlr/domains.html


def initialize_mdp_data(num_states):
    """Initialize uniform transitions, zero rewards, and small random values."""
    transition_counts = np.zeros((num_states, num_states, 2))
    transition_probs = np.ones((num_states, num_states, 2)) / num_states
    # Columns: failure count, visit count.
    reward_counts = np.zeros((num_states, 2))
    reward = np.zeros(num_states)
    value = np.random.rand(num_states) * 0.1

    return {
        'transition_counts': transition_counts,
        'transition_probs': transition_probs,
        'reward_counts': reward_counts,
        'reward': reward,
        'value': value,
        'num_states': num_states,
    }


def choose_action(state, mdp_data):
    """Choose the action with the highest expected next-state value."""
    tr_probs = mdp_data['transition_probs'][state]
    value = mdp_data['value']

    expect_value = value.dot(tr_probs)
    return np.argmax(expect_value)


def update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, reward):
    """Record a transition and the reward at the destination state."""
    tr_counts = mdp_data['transition_counts']
    reward_counts = mdp_data['reward_counts']

    tr_counts[state, new_state, action] += 1
    if reward == -1: reward_counts[new_state, 0] += 1
    reward_counts[new_state, 1] += 1

    return


def update_mdp_transition_probs_reward(mdp_data):
    """Estimate transitions and rewards, leaving unseen states unchanged."""
    tr_counts = mdp_data['transition_counts']
    tr_probs = mdp_data['transition_probs']

    reward_counts = mdp_data['reward_counts']

    total = np.sum(tr_counts, axis=1, keepdims=True)
    np.divide(tr_counts, total, out=tr_probs, where=(total!=0))

    mask = reward_counts[:, 1]!=0
    mdp_data['reward'][mask] = - reward_counts[mask, 0] / reward_counts[mask, 1]

    return


def update_mdp_value(mdp_data, tolerance, gamma):
    """Run value iteration; return whether it converged in one iteration."""
    tr_probs = mdp_data['transition_probs']

    reward = mdp_data['reward']
    value = mdp_data['value']

    iters = 0
    while True:
        iters += 1

        value_new = reward + gamma * np.max(value.dot(tr_probs), axis=1)
        if np.max(np.abs(value_new-value)) < tolerance: break

        value = value_new

    mdp_data['value'] = value_new

    return iters == 1


def main(plot=True):
    seed = 2
    np.random.seed(seed)

    pause_time = 0.0001
    min_trial_length_to_start_display = 100
    display_started = min_trial_length_to_start_display == 0

    NUM_STATES = 163
    GAMMA = 0.995
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20

    time = 0

    time_steps_to_failure = []
    num_failures = 0
    time_at_start_of_current_trial = 0

    max_failures = 500

    cart_pole = CartPole(Physics())

    # Continuous state: position, velocity, angle, angular velocity.
    x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
    state_tuple = (x, x_dot, theta, theta_dot)

    state = cart_pole.get_state(state_tuple)

    mdp_data = initialize_mdp_data(NUM_STATES)

    consecutive_no_learning_trials = 0
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:
        action = choose_action(state, mdp_data)

        state_tuple = cart_pole.simulate(action, state_tuple)

        time = time + 1

        new_state = cart_pole.get_state(state_tuple)

        if new_state == NUM_STATES - 1:
            R = -1
        else:
            R = 0

        update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, R)

        # Update the model and value function after each failure.
        if new_state == NUM_STATES - 1:
            update_mdp_transition_probs_reward(mdp_data)

            converged_in_one_iteration = update_mdp_value(mdp_data, TOLERANCE, GAMMA)

            if converged_in_one_iteration:
                consecutive_no_learning_trials = consecutive_no_learning_trials + 1
            else:
                consecutive_no_learning_trials = 0

        if new_state == NUM_STATES - 1:
            num_failures += 1
            if num_failures >= max_failures:
                break
            print('[INFO] Failure number {}'.format(num_failures))
            time_steps_to_failure.append(time - time_at_start_of_current_trial)
            time_at_start_of_current_trial = time

            if time_steps_to_failure[num_failures - 1] > min_trial_length_to_start_display:
                display_started = 1

            x = -1.1 + np.random.uniform() * 2.2
            x_dot, theta, theta_dot = 0.0, 0.0, 0.0
            state_tuple = (x, x_dot, theta, theta_dot)
            state = cart_pole.get_state(state_tuple)
        else:
            state = new_state

    if plot:
        log_tstf = np.log(np.array(time_steps_to_failure))
        plt.plot(np.arange(len(time_steps_to_failure)), log_tstf, 'k')
        window = 30
        w = np.array([1/window for _ in range(window)])
        weights = lfilter(w, 1, log_tstf)
        x = np.arange(window//2, len(log_tstf) - window//2)
        plt.plot(x, weights[window:len(log_tstf)], 'r--')
        plt.xlabel('Num failures')
        plt.ylabel('Log of num steps to failure')
        plt.title('seed = {}'.format(seed))
        plt.savefig('output/control_{}.png'.format(seed))

    return np.array(time_steps_to_failure)


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)
    print(os.getcwd())
    main()
