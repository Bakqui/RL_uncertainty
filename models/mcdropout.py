import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from models.utils import ReplayBuffer, get_batch

class MCDropout(nn.Module):
    def __init__(self, in_dim, o_dim, h_dim,
                 h_act=nn.ELU, dropout=0.5,
                 noise_level=None, agent=True):
        super(MCDropout, self).__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        if noise_level is None:
            self.fc3 = nn.Linear(h_dim, 2*o_dim)
            self.noise_level = noise_level
        else:
            self.fc3 = nn.Linear(h_dim, o_dim)
            self.noise_level = noise_level
        self.h_act = h_act()
        self.p = dropout
        self.o_dim = o_dim
        self.agent = agent

    def forward(self, x):
        x = self.h_act(self.fc1(x))
        x = F.dropout(x, p=self.p, training=self.agent)
        x = self.h_act(self.fc2(x))
        x = F.dropout(x, p=self.p, training=self.agent)
        if self.noise_level is None:
            mu, var = torch.split(self.fc3(x), self.o_dim, dim=1)
            var = F.softplus(var)
            return mu, var
        else:
            return self.fc3(x), self.noise_level

    def sample(self, input, samples):
        outputs = torch.zeros(samples, input.shape[0], self.o_dim)
        if self.noise_level is None:
            uns = torch.zeros(samples, input.shape[0], self.o_dim)
        for i in range(samples):
            if self.noise_level is None:
                mu, var = self(input)
                outputs[i] = mu
                uns[i] = var.sqrt()
            else:
                outputs[i] = self(input)[0]
        mu = outputs.mean(0)
        un_model = outputs.std(0)
        if self.noise_level is None:
            un_noise = uns.mean(0)
        else:
            un_noise = 0
        return mu, un_model+un_noise

class DropDQNAgent(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim,
                 h_act=nn.ReLU, buffer_size=100000,
                 batch_size=32, lr=1e-4, gamma=0.95,
                 theta=0.01, dropout=0.5, weight_decay=0.1,
                 noise_level=None, n_sample=5,
                 *args, **kwargs):
        super(DropDQNAgent, self).__init__()
        self.q_net = MCDropout(in_dim=s_dim, o_dim=a_dim, h_dim=h_dim,
                               h_act=h_act, dropout=dropout,
                               noise_level=noise_level)
        self.target_net = MCDropout(in_dim=s_dim, o_dim=a_dim, h_dim=h_dim,
                                    h_act=h_act, dropout=dropout,
                                    noise_level=noise_level, agent=False)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizer = Adam(self.q_net.parameters(), lr=lr,
                              weight_decay=weight_decay)
        self.gamma = gamma
        self.theta = theta

        self.noise_level = noise_level
        self.n_sample = n_sample
        self.a_dim = a_dim

    # T.B.U.
    def forward(self, x):
        if self.training:
            self.q_net.agent = True
            return self.q_net.sample(x, self.n_sample)
        else:
            self.q_net.agent = False
            return self.q_net(x)

    def save_memory(self, ex):
        self.buffer.push(ex)

    # T.B.U.
    def train(self, k=1, max_norm=None):
        losses = []
        self.q_net.agent = True
        for _ in range(k):
            experiences = self.buffer.sample(self.batch_size)
            s, a, r, t, mask = get_batch(experiences)
            next_mu, _ = self.target_net(t)
            next_q = next_mu.max(-1, keepdim=True)[0]
            target = r + self.gamma*mask*next_q.detach()
            pred_mu, pred_un = self.q_net.sample(s, self.n_sample)
            pred = pred_mu.gather(-1, a)
            un = pred_un.gather(-1, a)
            loss = -1. * D.Normal(pred, un).log_prob(target).mean()
            self.optimizer.zero_grad()
            loss.backward()
            if max_norm is not None:
                clip_grad_norm_(self.q_net.parameters(), max_norm)
            self.optimizer.step()
            losses.append(loss.item())
        self.target_update()
        return np.mean(losses)

    def train_start(self):
        return (len(self.buffer) >= self.batch_size)

    def target_update(self):
        for target, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target.data = (1-self.theta)*target.data + self.theta*param.data
