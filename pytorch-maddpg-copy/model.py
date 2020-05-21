import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent
        hide = 32#1024
        hide2 = 8#512
        hide3 = 4#300
        self.FC1 = nn.Linear(obs_dim, hide)
        self.FC2 = nn.Linear(hide+act_dim, hide2)
        self.FC3 = nn.Linear(hide2, hide3)
        self.FC4 = nn.Linear(hide3, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        hide = 8#500
        hide2 = 4#128
        self.FC1 = nn.Linear(dim_observation, hide)
        self.FC2 = nn.Linear(hide, hide2)
        self.FC3 = nn.Linear(hide2, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        result = F.softmax(result)
        #result = F.softmax(result)
        return result