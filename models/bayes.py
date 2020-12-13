import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from models.core import BayesLinear
from models.utils import ReplayBuffer, get_batch

class BayesNet(nn.Module):
    def __init__(self, in_dim, o_dim, h_dim,
                 h_act=nn.ELU, noise_level=None):
        super(BayesNet, self).__init__()
        self.fc1 = BayesLinear(in_dim, h_dim)
        self.fc2 = BayesLinear(h_dim, h_dim)
        if noise_level is None:
            self.fc3 = BayesLinear(h_dim, 2*o_dim)
            self.noise_level = noise_level
        else:
            self.fc3 = BayesLinear(h_dim, o_dim)
            self.noise_level = noise_level
        self.h_act = h_act()
        self.o_dim = o_dim

    def forward(self, x):
        x = self.h_act(self.fc1(x))
        x = self.h_act(self.fc2(x))
        if self.noise_level is None:
            mu, var = torch.split(self.fc3(x), self.o_dim, dim=1)
            var = F.softplus(var)
            return mu, var
        else:
            return self.fc3(x), self.noise_level

    def log_p(self):
        return self.fc1.log_p + self.fc2.log_p + self.fc3.log_p

    def log_q(self):
        return self.fc1.log_q + self.fc2.log_q + self.fc3.log_q

    def sample(self, input, samples, kl=False):
        outputs = torch.zeros(samples, input.shape[0], self.o_dim)
        if self.noise_level is None:
            uns = torch.zeros(samples, input.shape[0], self.o_dim)
        if kl:
            log_priors = torch.zeros(samples)
            log_var_posts = torch.zeros(samples)
        for i in range(samples):
            if self.noise_level is None:
                mu, var = self(input)
                outputs[i] = mu
                uns[i] = var.sqrt()
            else:
                outputs[i] = self(input)[0]
            if kl:
                log_priors[i] = self.log_p()
                log_var_posts[i] = self.log_q()
        mu = outputs.mean(0)
        un_model = outputs.std(0)
        if self.noise_level is None:
            un_noise = uns.mean(0)
        else:
            un_noise = 0
        if kl:
            return mu, un_model+un_noise, log_priors.mean(), log_var_posts.mean()
        return mu, un_model+un_noise

class B3DQNAgent(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim,
                 h_act=nn.ReLU, buffer_size=100000,
                 batch_size=32, lr=1e-4, gamma=0.95,
                 theta=0.01, noise_level=None,
                 n_sample=5, *args, **kwargs):
        super(B3DQNAgent, self).__init__()
        self.q_net = BayesNet(in_dim=s_dim, o_dim=a_dim,
                              h_dim=h_dim, h_act=h_act,
                              noise_level=noise_level)
        self.target_net = BayesNet(in_dim=s_dim, o_dim=a_dim,
                                   h_dim=h_dim, h_act=h_act,
                                   noise_level=noise_level)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.theta = theta

        self.noise_level = noise_level
        self.n_sample = n_sample
        self.a_dim = a_dim

    def forward(self, x):
        if self.training:
            self.q_net.train()
            return self.q_net.sample(x, self.n_sample)
        else:
            self.q_net.eval()
            return self.q_net(x)

    def save_memory(self, ex):
        self.buffer.push(ex)

    def train(self, lamb, k=1, max_norm=None):
        losses = []
        self.q_net.train()
        for _ in range(k):
            experiences = self.buffer.sample(self.batch_size)
            s, a, r, t, mask = get_batch(experiences)
            self.target_net.eval()
            next_mu, _ = self.target_net(t)
            next_q = next_mu.max(-1, keepdim=True)[0]
            target = r + self.gamma*mask*next_q.detach()
            preds, uns, log_prior, log_var_post = self.q_net.sample(s, self.n_sample, True)
            pred = preds.gather(-1, a)
            un = uns.gather(-1, a)
            ll = D.Normal(pred, un).log_prob(target).mean()
            loss = lamb*(log_var_post - log_prior) - ll
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
        self.target_net.train()
        for target, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target.data = (1-self.theta)*target.data + self.theta*param.data
