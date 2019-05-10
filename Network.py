import pdb
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        obs_dim = dim_observation * n_agent
        self.act_dim = dim_action * n_agent

        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024 + self.act_dim, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 300),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(300, 1)

    # obs:batch_size * obs_dim
    def forward(self, obs, acts, fla="none"):
        x = self.fc1(obs )
        combined = torch.cat([x, acts ], 1)
        if fla=="actor":
            combined = torch.cat([x.detach(), acts], 1)

        x = self.fc2(combined )
        x = self.fc3(x)
        return self.fc4(x)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_observation, 500),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500, 128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, dim_action),
            nn.Tanh()
        )

    def forward(self, obs):
        obs = torch.FloatTensor(obs )
        x = self.fc1(obs); x = self.fc2(x)
        return self.fc3(x)