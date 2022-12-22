from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from cs285.agents.base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
from cs285.infrastructure.sac_utils import soft_update_params
import cs285.infrastructure.pytorch_util as ptu
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: DONE...?
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic 

        # sample action
        next_ac_na, log_probs = self.actor.get_action(next_ob_no, sample=True)
        
        # convert to tensor
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        next_ac_na = ptu.from_numpy(next_ac_na)
        ac_na = ptu.from_numpy(ac_na)
        terminal_n = ptu.from_numpy(terminal_n)
        re_n = ptu.from_numpy(re_n)
        
        re_n = re_n.unsqueeze(1)
        terminal_n = terminal_n.unsqueeze(1)
        
        # collect target critic q values
        next_q_estimates = torch.stack(self.critic_target(next_ob_no, next_ac_na)).min(dim=0)[0]

        # collect critic q values
        q1, q2 = self.critic(ob_no, ac_na)
        y_n = re_n + self.gamma * (1 - terminal_n) * (next_q_estimates - self.actor.alpha * log_probs)
        # optimize critic
        l1 = self.critic.loss(q1, y_n.detach())
        l2 = self.critic.loss(q2, y_n.detach())
        critic_loss = l1 + l2 

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        if torch.any(torch.isnan(self.critic.Q1[0].weight)):
          import pdb;pdb.set_trace()

        return critic_loss.item()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging

        loss = OrderedDict()
        # update critic
        critic_loss = []
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
          critic_loss.append(self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n))
        # update target
        if self.training_step % self.critic_target_update_frequency == 0:
          soft_update_params(self.critic, self.critic_target, self.critic_tau)

        # update actor
        actor_loss, alpha_loss = [], []
        if self.training_step % self.actor_update_frequency == 0:
          for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            act_loss, alph_loss, alpha = self.actor.update(ob_no, self.critic)
            actor_loss.append(act_loss)
            alpha_loss.append(alph_loss)
        
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)