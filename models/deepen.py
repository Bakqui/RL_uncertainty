import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from models.core import H_MLP
from models.utils import ReplayBuffer, get_batch


class EnModel(nn.Module):
    def __init__(self, model=H_MLP, n_model=5, *args, **kwargs):
        super(EnModel, self).__init__()
        self.n_model = n_model
        self.heads = nn.ModuleList()
        for i in range(n_model):
            self.heads.append(model(*args, **kwargs))

    def forward(self, x):
        mu_list, var_list = [], []
        for head in self.heads:
            mu, var = head(x)
            mu_list.append(mu)
            var_list.append(var)
        mu_list = torch.stack(mu_list, dim=-1).squeeze()
        var_list = torch.stack(var_list, dim=-1).squeeze()
        mu_n = torch.mean(mu_list, axis=-1, keepdim=True)
        var_n = torch.mean(torch.pow(mu_list, 2)+var_list,
                           axis=-1, keepdim=True) - torch.pow(mu_n, 2)
        return mu_n, var_n

class EnDQNAgent(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim,
                 h_act=nn.ReLU, buffer_size=100000,
                 batch_size=32, lr=1e-4, gamma=0.95,
                 theta=0.01, n_model=5,
                 *args, **kwargs):
        super(EnDQNAgent, self).__init__()
        self.q_nets = EnModel(in_dim=s_dim, o_dim=a_dim, h_dim=h_dim,
                              h_act=h_act, n_model=5)
        self.target_nets = EnModel(in_dim=s_dim, o_dim=a_dim, h_dim=h_dim,
                                   h_act=h_act, n_model=5)
        self.target_nets.load_state_dict(self.q_nets.state_dict())
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        self.optimizers = [Adam(head.parameters(), lr=lr)
                           for head
                           in self.q_nets.heads]
        self.gamma = gamma
        self.theta = theta
        self.n_model = n_model
        self.a_dim = a_dim

    # T.B.U.
    def forward(self, x):
        return self.q_nets(x)

    def save_memory(self, ex):
        self.buffer.push(ex)

    # T.B.U.
    def train(self, k=1, max_norm=None):
        losses = []
        for _ in range(k):
            for m in range(self.n_model):
                q_net = self.q_nets.heads[m]
                target_net = self.target_nets.heads[m]
                optimizer = self.optimizers[m]
                experiences = self.buffer.sample(self.batch_size)
                s, a, r, t, mask = get_batch(experiences)
                next_q_mu, _ = target_net(t)
                next_q = next_q_mu.max(-1, keepdim=True)[0]
                target = r + self.gamma*mask*next_q.detach()
                pred_mu, pred_var = q_net(s)
                pred = pred_mu.gather(-1, a)
                un = pred_var.sqrt().gather(-1, a)
                loss = -1. * D.Normal(pred, un).log_prob(target).mean()
                optimizer.zero_grad()
                loss.backward()
                if max_norm is not None:
                    clip_grad_norm_(q_net.parameters(), max_norm)
                optimizer.step()
                losses.append(loss.item())
        self.target_update()
        return np.mean(losses)

    def train_start(self):
        return (len(self.buffer) >= self.batch_size)

    def target_update(self):
        for target, param in zip(self.target_nets.parameters(), self.q_nets.parameters()):
            target.data = (1-self.theta)*target.data + self.theta*param.data
