import gradio as gr
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize environment and model
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
dqn = DQN(state_dim, action_dim)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
criterion = nn.MSELoss()

def run_episode(learning_rate, epsilon):
    optimizer.param_groups[0]["lr"] = learning_rate
    state, _ = env.reset()
    total_reward = 0
    steps = []

    for _ in range(200):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(dqn(state_tensor)).item()

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps.append(state[0])
        state = next_state

        if done:
            break

    # Generate plot
    plt.figure(figsize=(5, 3))
    plt.plot(steps, label="State[0] Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    plt.legend()
    plt.tight_layout()
    plot_path = "plot.png"
    plt.savefig(plot_path)
    plt.close()

    return total_reward, plot_path

iface = gr.Interface(
    fn=run_episode,
    inputs=[
        gr.Slider(0.0001, 0.01, value=0.001, label="Learning Rate"),
        gr.Slider(0, 1, value=0.1, label="Epsilon")
    ],
    outputs=["number", gr.Image(type="filepath")],  # Fixing output display
    title="Interactive RL Agent",
    description="Run an RL agent on CartPole and tweak its learning parameters."
)

if __name__ == "__main__":
    iface.launch()

