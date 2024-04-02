import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from replaybuffer import ReplayBuffer
from cartpole_env import CartPole2DEnv
from dqn import DQN


def preprocess_state(state):
    try:
        # Attempt to directly convert and flatten the state
        return np.array(state, dtype=np.float32).flatten()
    except ValueError:
        # Handle nested sequences or varying lengths; adjust based on actual state structure
        # This is a generic handler, might need adjustment
        flattened = [item for sublist in state for item in sublist]
        return np.array(flattened, dtype=np.float32)


def main():
    env = CartPole2DEnv(render_mode='human')
    input_size = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.observation_space.n
    model = DQN(input_size, env.action_space.n)
    target_model = DQN(input_size, env.action_space.n)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(10000)

    BATCH_SIZE = 64
    GAMMA = 0.99
    TARGET_UPDATE = 10
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = 200
    epsilon = epsilon_start

    episode_rewards = []

    for episode in range(500):
        state = env.reset()
        state = preprocess_state(state)  # Preprocess the state right after retrieval

        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.from_numpy(state).unsqueeze(0).float()
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(state_tensor).max(1)[1].view(1, 1).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_state(next_state)  # Preprocess next_state

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state  # Update the state with the preprocessed next_state
            total_reward += reward

            if len(replay_buffer) >= BATCH_SIZE:
                experiences = replay_buffer.sample(BATCH_SIZE)
                batch = map(np.array, zip(*experiences))
                states, actions, rewards, next_states, dones = [torch.tensor(data) for data in batch]

                # Model update logic remains the same...

            # Epsilon decay and target model update logic...

        episode_rewards.append(total_reward)
        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode}, Total Reward: {total_reward}")
    plt.plot(episode_rewards)
    plt.title('Episode vs Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


if __name__ == "__main__":
    main()
