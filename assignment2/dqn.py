import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_size, action_space_n):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 130)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(130, action_space_n)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
