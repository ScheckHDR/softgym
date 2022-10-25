'''
https://github.com/polixir/OfflineRL/blob/3ef71477daa7c165996b00dcdf89716f2b22b8d8/offlinerl/algo/modelfree/cql.py#L102
used to base this file off of.
'''

import copy
import numpy as np
from typing import List, Union, Dict, Optional
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Normal

class CustExtractor:

    def __init__(self, sample_obs ,out_dim: int = 512):

        self.cnn = nn.Sequential(
            # nn.Identity()
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # layers = [nn.Identity()]

        # Compute shape by doing one forward pass
        with torch.no_grad():
            test = torch.as_tensor(sample_obs['cross'].sample()).unsqueeze(0).float()
            n_flatten = self.cnn(test).shape[1] + np.prod(sample_obs['tail'].shape) + np.prod(sample_obs['shape'].shape)

        self.linear = nn.Sequential(nn.Linear(n_flatten, out_dim), nn.ReLU())
        
        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_result = self.cnn(observations['cross'])
        # print(observations['cross'].shape)
        # print(f'tail:{observations["tail"].flatten(1).shape}')
        # print(f'shape: {observations["shape"].flatten(1).shape}')
        # print(f'cnn: {cnn_result.shape}')
        lin_input = torch.concat([cnn_result,observations['tail'].flatten(1),observations['shape'].flatten(1)],dim=1)
        return self.linear(lin_input)

class Critic(nn.Module):
    def __init__(
        self,
        state_extractor,
        network:nn.Module
    ):
        super().__init__()
        self.extractor = state_extractor
        self.Q = network

    def forward(
        self,
        state,
        action,
    ):
        logits = self.extractor(state)

        logits_a = torch.cat([logits,action],dim=1)
        return self.Q(logits_a)

class ActorProb(nn.Module):
    def __init__(
        self,
        state_extractor,
        network:Optional[nn.Module] = None
    ):
        super().__init__()
        if network is not None:
            self.net = torch.cat([state_extractor,network])
        else:
            self.net = state_extractor

    def forward(self,obs) -> Normal:
        logits = self.net(obs)

        # TODO: Something about an optional conditioned sigma.

        shape=[1]*len(logits.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(logits)).exp()

        return Normal(logits,sigma)

    def log_prob(self,obs,actions):
        normal = self(obs)
        log_prob = normal.log_prob(actions)
        return log_prob.sum(-1)

class MLP(nn.Module):
    def __init__(self):
        pass
    


class CQLTrainer:
    def __init__(
        self,
        actor_kwargs:Dict,
        critic_kwargs:List[Dict], # List so that we can have multiple critics.
    ):
        if isinstance(critic_kwargs,Dict):
            critic_kwargs = [critic_kwargs]

        self.critics = []
        self.target_critics = []
        for args in critic_kwargs:
            extractor = CustExtractor()
            net = MLP(**args)
            critic = Critic(extractor,net)
            self.critics.append(critic)
            self.target_critics.append(copy.deepcopy(critic))

        actor_extractor = CustExtractor()
        net = MLP(**actor_kwargs)
        self.actor = ActorProb(actor_extractor,net)

    def forward(self,obs,reparam=True,return_log_prob=True):
        normal = self.actor(obs)
        if reparam:
            action = normal.rsample()
        else:
            action = normal.sample()
            
        log_prob = None
        if return_log_prob:
            log_prob = normal.log_prob(action)
            log_prob = log_prob.sum(dim=1,keepdim=True)

        return action,log_prob



    def _train_mini_batch(self,batch:pd.DataFrame):
        
        batch = torch.tensor(batch.values)

        rewards = batch.reward
        terminals = batch.done
        obs = batch.obs
        actions = batch.action
        next_obs = batch.next_obs

        new_obs_action,log_pi = self.forward(obs)
        # Policy and alpha loss
        
        # Alpha set it to 1 to make everything work. TODO: add entropy stuff.
        alpha_loss = 0
        alpha = 1


        # Behavioural Cloning. TODO: add non-BC later.
        policy_log_prob = self.actor.log_prob(obs,actions)
        policy_loss = (alpha*log_pi-policy_log_prob).mean()

        # Q-function loss.
        q_preds = [critic(obs,actions) for critic in self.critics]
        new_curr_action, new_curr_log_pi = self.forward(obs,reparam=True,return_log_prob=True)
        new_next_actions, new_log_pi = self.forward(next_obs,reparam=True,return_log_prob=True)

        # Using max q backup. TODO: Other options.
        next_actions_temp,_ = self._get_policy_actions(next_obs,num_actions=10,network=self.forward)
        target_Q_values = [self._get_tensor_values(next_obs,next_actions_temp,network=critic).max(1)[0].view(-1,1) for critic in self.critics]
        target_Q_values = torch.min(*target_Q_values)

        q_target = rewards + (1-terminals) * self.gamma * target_Q_values.detach()

        q_losses = [self.critic_criterion(pred,q_target) for pred in q_preds]

        # Add CQL.
        rand_act = torch.FloatTensor(q_preds[0].shape[0]*10,actions.shape[-1]).uniform_(-1,1).to(self.device)
        curr_act_tens,curr_log_pis = self._get_policy_actions(obs,10,network=self.forward)
        new_curr_act_tens,new_log_pis = self._get_policy_actions(next_obs,10,network=self.forward)
        q_rand = [self._get_tensor_values(obs,rand_act,network=critic) for critic in self.critics]
        q_curr_actions = [self.obs(curr_act_tens,network=critic) for critic in self.critics]
        q_next_actions = [self._get_tensor_values(obs,new_curr_act_tens,network=critic) for critic in self.critics]

        q_cats = [torch.cat([q_rand[i],q_preds[i].unsqueeze(1),q_next_actions[i],q_curr_actions[i]]) for i in range(len(self.critics))]

        # TODO: Something about importance sampling.


        min_q_losses = [(torch.logsumexp(q_cats[i]/self.temperature,dim=1).mean()*self.temperature - q_preds[i])*self.min_q_weight for i in range(len(self.critics))]


        # TODO: Something about Lagrange thresh.


        q_losses = [self.explore*q_losses[i] + (2-self.explore)*min_q_losses[i] for i in range(len(self.critics))]

        for i in range(len(self.critics)):
            self.critic_optim[i].zero_grad()
            q_losses[i].backward(retain_graph=True)
            self.critic_optim[i].step()

            self._sync_weight(self.critic_targets[i],self.critic[i],self.soft_target_tau)

        self._n_train_steps_total += 1

