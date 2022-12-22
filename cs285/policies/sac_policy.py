from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate
        self.sac_mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=2*self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
        self.sac_mean_net.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain(self.sac_mean_net.parameters()),
            self.learning_rate
        )

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        obs = ptu.from_numpy(obs)
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        action_dist, action = self(observation)

        if not sample:
          action = action_dist.mean
        
        log_prob = action_dist.log_prob(action)
        action = ptu.to_numpy(action)
        assert action.shape[-1] == self.ac_dim
        return action, log_prob

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        # from HW2
        action_props = self.sac_mean_net(observation)
        mean = action_props[:, :self.ac_dim]
        log_std = action_props[:, self.ac_dim:]
        log_scale = torch.clamp(log_std.tanh(), self.log_std_bounds[0] + 1e-6, self.log_std_bounds[1] - 1e-6)
        scale = torch.exp(log_scale)

        action_dist = sac_utils.SquashedNormal(mean, scale)
        action = torch.clamp(action_dist.rsample(), self.action_range[0] + 1e-6, self.action_range[1] - 1e-6)
        return action_dist, action

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        
        sampled_actions, log_prob = self.get_action(obs, sample=True)
        sampled_actions = ptu.from_numpy(sampled_actions)
        obs = ptu.from_numpy(obs)
        pred_q = torch.stack(critic(obs, sampled_actions)).min(dim=0)[0]
        # actor loss
        actor_loss = ((self.alpha.detach() * log_prob) - pred_q).mean()
       
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # alpha loss
        alpha_loss = -self.alpha * (log_prob.detach() + self.target_entropy)
        alpha_loss = alpha_loss.mean()
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha

