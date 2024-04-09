import pickle
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def sarsa_update(q, learning_rate, reward, next_q_value, discount_factor):
    return q + learning_rate * (reward + discount_factor * next_q_value - q)


def q_learning_update(q, state, reward, new_state, learning_rate, discount_factor):
    return q[state] + learning_rate * (reward + discount_factor * np.max(q[new_state]) - q[state])


def run(is_training=True, render=False, algorithm='q_learning'):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Divide position, velocity, pole angle, and pole angular velocity into segments
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if is_training:
        q = np.zeros((len(pos_space) + 1, len(vel_space) + 1, len(ang_space) + 1,
                      len(ang_vel_space) + 1, env.action_space.n))  # init a 11x11x11x11x2 array
    else:
        f = open('cartpole.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.1  # alpha or learning rate
    discount_factor_g = 0.99  # gamma or discount factor.

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.00001  # epsilon decay rate
    rng = np.random.default_rng()  # random number generator

    rewards_per_episode = []

    i = 0

    # for i in range(episodes):
    while True:
        state = env.reset()[0]  # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False  # True when reached goal
        rewards = 0

        if algorithm == 'sarsa':
            # Initial action for SARSA
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

        while not terminated and rewards < 10000:
            if algorithm == 'q_learning' or not is_training:
                # Select action using epsilon-greedy policy
                if is_training and rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if algorithm == 'sarsa':
                # Choose the next action using the same policy for SARSA
                if is_training and rng.random() < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(
                        q[new_state_p, new_state_v, new_state_a, new_state_av, :]
                    )

            # Update Q-values using the selected algorithm
            if is_training:
                if algorithm == 'sarsa':
                    # Assuming next_action has been chosen correctly
                    next_q_value = q[
                        new_state_p, new_state_v, new_state_a, new_state_av, next_action
                    ]

                    q[state_p, state_v, state_a, state_av, action] = sarsa_update(
                        q=q[state_p, state_v, state_a, state_av, action],
                        learning_rate=learning_rate_a,
                        reward=reward,
                        next_q_value=next_q_value,
                        discount_factor=discount_factor_g
                    )
                    action = next_action  # Update action for the next iteration
                elif algorithm == 'q_learning':
                    # Use the Q-Learning update rule
                    q[state_p, state_v, state_a, state_av, action] = q_learning_update(
                        q, (state_p, state_v, state_a, state_av, action), reward,
                        (new_state_p, new_state_v, new_state_a, new_state_av),
                        learning_rate_a, discount_factor_g
                    )

            state, state_p, state_v, state_a, state_av = (new_state,
                                                          new_state_p,
                                                          new_state_v,
                                                          new_state_a,
                                                          new_state_av
                                                          )
            rewards += reward

            if not is_training and rewards % 100 == 0:
                print(f'Episode: {i}  Rewards: {rewards}')

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[max(0, len(rewards_per_episode) - 100):])

        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  '
                  f'Epsilon: {epsilon:0.2f}  '
                  f'Mean Rewards {mean_rewards:0.1f}')

        if mean_rewards > 1000:  # Threshold to consider the task solved or for early stopping
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)  # Update epsilon according to its decay rate

        i += 1  # Increment episode counter

    env.close()

    # Save Q table to file
    if is_training:
        f = open('cartpole.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]))
    plt.plot(mean_rewards)
    plt.savefig('cartpole.png')


if __name__ == '__main__':
    run(is_training=True, render=False, algorithm='q_learning')
    #run(is_training=False, render=True, algorithm='q_learning')
