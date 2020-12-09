import torch
from torch import nn
import torch.distributions as distributions

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh')/3)
        nn.init.normal_(m.bias, mean=0, std=0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.1):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs)
        )
        self.log_std = nn.Parameter(torch.log(torch.ones(1, num_outputs) * std))
        
        self.critic.apply(init_weights)
        self.actor.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = distributions.normal.Normal(mu, std)
        return dist, value
    
    def act(self, x):
        mu = self.actor(x)

        # # In eval mode, return mu directly
        # if not self.training:
        #     return mu

        std = self.log_std.exp().expand_as(mu)
        dist = distributions.normal.Normal(mu, std)
        return dist
