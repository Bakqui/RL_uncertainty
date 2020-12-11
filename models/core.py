import math
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

class DiagGaussian:
    def __init__(self, mu, rho):
        super(DiagGaussian, self).__init__()
        self.loc = mu
        self.rho = rho
        self.dist = D.Normal(0, 1)

    @property
    def scale(self):
        return F.softplus(self.rho)

    def log_prob(self, input: torch.Tensor) -> torch.Tensor:
        rst = -math.log(math.sqrt(2 * math.pi))
        rst -= torch.log(self.scale)
        rst -= ((input - self.loc) ** 2) / (2 * self.scale ** 2)
        return rst.sum()

    def rsample(self):
        eps = self.dist.sample(self.rho.size())
        return self.loc + self.scale*eps

    def __repr__(self):
        return self.__class__.__name__ +\
            '(loc: {}, scale: {})'.format(
                self.loc.size(), self.scale.size()
            )

class ScaleMixture:
    def __init__(self, pi=0.5, scale1=torch.Tensor([1.]),
                 scale2=torch.Tensor([0.0025])):
        self.pi = pi
        self.scale1 = scale1
        self.scale2 = scale2
        self.comp1 = D.Normal(0, scale1)
        self.comp2 = D.Normal(0, scale2)

    def log_prob(self, input: torch.Tensor) -> torch.Tensor:
        prob1 = torch.exp(self.comp1.log_prob(input))
        prob2 = torch.exp(self.comp2.log_prob(input))
        return (torch.log(self.pi*prob1 + (1-self.pi)*prob2)).sum()

class BayesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_fetures = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = bias
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.weight = DiagGaussian(self.weight_mu, self.weight_rho)
        self.weight_prior = ScaleMixture()
        if bias:
            self.bias = DiagGaussian(self.bias_mu, self.bias_rho)
            self.bias_prior = ScaleMixture()
        else:
            self.bias = None
            self.bias_prior = None
        self.log_p = 0
        self.log_q = 0

    def reset_parameters(self):
        nn.init.uniform_(self.weight_mu, -0.2, 0.2)
        nn.init.uniform_(self.weight_rho, -5., -4.)
        if self.bias:
            nn.init.uniform_(self.bias_mu, -0.2, 0.2)
            nn.init.uniform_(self.bias_rho, -5, -4)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight.rsample()
            self.log_p = self.weight_prior.log_prob(weight)
            self.log_q = self.weight.log_prob(weight)
            if self.bias is not None:
                bias = self.bias.rsample()
                self.log_p = self.log_p + self.bias_prior.log_prob(bias)
                self.log_q = self.log_q + self.bias.log_prob(bias)
            else:
                bias = None
        else:
            weight = self.weight.loc
            self.log_p = 0
            self.log_q = 0
            if self.bias is not None:
                bias = self.bias.loc
            else:
                bias = None
        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_fetures, self.bias is not None
        )

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, o_dim, h_dim,
                 h_act=nn.ELU):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, o_dim)
        self.h_act = h_act()

    def forward(self, x):
        x = self.h_act(self.fc1(x))
        x = self.h_act(self.fc2(x))
        return self.fc3(x)

class H_MLP(nn.Module):
    def __init__(self, in_dim, o_dim, h_dim,
                 h_act=nn.ELU):
        super(H_MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 2*o_dim)

        self.h_act = h_act()
        self.o_dim = o_dim

    def forward(self, x):
        x = self.h_act(self.fc1(x))
        x = self.h_act(self.fc2(x))
        mu, var = x = torch.split(self.fc3(x), self.o_dim, dim=1)
        var = F.softplus(var)
        return mu, var
