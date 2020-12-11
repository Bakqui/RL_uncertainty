import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from models.core import SimpleMLP
from models.utils import ReplayBuffer, get_batch


class DQNAgent(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim,
                 h_act=nn.ReLU, buffer_size=100000,
                 batch_size=32, lr=1e-4, gamma=0.95,
                 theta=0.01, *args, **kwargs):
        super(DQNAgent, self).__init__()
        self.q_net = SimpleMLP(in_dim=s_dim, o_dim=a_dim,
                               h_dim=h_dim, h_act=h_act)
        self.target_net = SimpleMLP(in_dim=s_dim, o_dim=a_dim,
                                    h_dim=h_dim, h_act=h_act)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.theta = theta
        self.a_dim = a_dim

    def forward(self, x):
        return self.q_net(x)

    def save_memory(self, ex):
        self.buffer.push(ex)

    def train(self, k=1, max_norm=None):
        losses = []
        for _ in range(k):
            experiences = self.buffer.sample(self.batch_size)
            s, a, r, t, mask = get_batch(experiences)
            next_q = self.target_net(t).max(-1, keepdim=True)[0]
            target = r + self.gamma*mask*next_q.detach()
            pred = self.q_net(s).gather(-1, a)
            loss = F.mse_loss(pred, target)
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

class BootDQNAgent(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim,
                 h_act=nn.ReLU, buffer_size=100000,
                 batch_size=32, lr=1e-4, gamma=0.95,
                 theta=0.01, n_model=5, *args, **kwargs):
        super(BootDQNAgent, self).__init__()
        q_list = [SimpleMLP(in_dim=s_dim, o_dim=a_dim,
                            h_dim=h_dim, h_act=h_act)
                  for _ in range(n_model)]
        target_list = [SimpleMLP(in_dim=s_dim, o_dim=a_dim,
                                 h_dim=h_dim, h_act=h_act)
                       for _ in range(n_model)]
        self.q_nets = nn.ModuleList(q_list)
        self.target_nets = nn.ModuleList(target_list)
        self.target_nets.load_state_dict(self.q_nets.state_dict())
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizers = [Adam(q_net.parameters(), lr=lr)
                           for q_net in self.q_nets]
        self.gamma = gamma
        self.theta = theta
        self.n_model = n_model
        self.current_head = None
        self.a_dim = a_dim

    def forward(self, x):
        if self.training:
            return self.q_nets[self.current_head](x)
        else:
            return torch.cat([q_net(x) for q_net in self.q_nets], dim=0)

    def save_memory(self, ex):
        self.buffer.push(ex)

    def train(self, k=1, max_norm=None):
        losses = []
        for _ in range(k):
            for m, q_net in enumerate(self.q_nets):
                experiences = self.buffer.sample(self.batch_size)
                s, a, r, t, mask = get_batch(experiences)
                next_q = self.target_nets[m](t).max(-1, keepdim=True)[0]
                target = r + self.gamma*mask*next_q.detach()
                pred = q_net(s).gather(-1, a)
                loss = F.mse_loss(pred, target)
                self.optimizers[m].zero_grad()
                loss.backward()
                if max_norm is not None:
                    clip_grad_norm_(q_net.parameters(), max_norm)
                self.optimizers[m].step()
                losses.append(loss.item())
        self.target_update()
        return np.mean(losses)

    def train_start(self):
        return (len(self.buffer) >= self.batch_size)

    def target_update(self):
        for target, param in zip(self.target_nets.parameters(), self.q_nets.parameters()):
            target.data = (1-self.theta)*target.data + self.theta*param.data
