'''
https://github.com/polixir/OfflineRL/blob/3ef71477daa7c165996b00dcdf89716f2b22b8d8/offlinerl/algo/modelfree/cql.py#L102
used to base this file off of.
'''

import copy
import numpy as np
from typing import Callable, List, Union, Dict, Optional
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data.dataloader import DataLoader




class CustExtractor(nn.Module):

    def __init__(self, sample_obs ,out_dim: int = 512,*args,**kwargs):
        super().__init__()

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
            test = torch.as_tensor(sample_obs['cross']).unsqueeze(0).unsqueeze(0).float()
            n_flatten = self.cnn(test).shape[1] + np.prod(sample_obs['tail'].shape) + np.prod(sample_obs['shape'].shape)

        self.linear = nn.Sequential(nn.Linear(n_flatten, out_dim), nn.ReLU())
        
        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_result = self.cnn(observations['cross'].unsqueeze(1).float())
        # print(observations['cross'].shape)
        # print(f'tail:{observations["tail"].flatten(1).shape}')
        # print(f'shape: {observations["shape"].flatten(1).shape}')
        # print(f'cnn: {cnn_result.shape}')
        lin_input = torch.cat([cnn_result,observations['tail'].flatten(1).float(),observations['shape'].flatten(1).float()],dim=1)
        return self.linear(lin_input)

    def __get_item__(self,key):
        return (self.cnn[:] + self.linear[:])[key]

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

        logits_a = torch.cat([logits,action],dim=1).float()
        return self.Q(logits_a)

    def __get_item__(self,key):
        return (self.extractor[:] + self.Q[:])[key]

class ActorProb(nn.Module):
    def __init__(
        self,
        state_extractor,
        network:Optional[nn.Module] = None
    ):
        super().__init__()
        self.extractor = state_extractor
        self.net = network

        self.sigma = nn.Parameter(torch.zeros(1,network[-2].out_features))

    def forward(self,obs) -> Normal:
        s = self.extractor(obs)
        logits = self.net(s)

        # TODO: Something about an optional conditioned sigma.

        shape=[1]*len(logits.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(logits)).exp()

        return Normal(logits,sigma)

    def log_prob(self,obs,actions):
        normal = self(obs)
        log_prob = normal.log_prob(actions)
        return log_prob.sum(-1)

    def __getitem__(self,key):
        return (self.extractor[:] + self.network[:])[key]

class MLP(nn.Module):
    def __init__(self,layer_sizes,mid_activation_func=nn.ReLU,final_activation_func=nn.Identity,*args,**kwargs):
        super().__init__()
        assert len(layer_sizes) > 1, f'Number of layers (including input and output) must be at least 2.'
         
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.extend([nn.Linear(layer_sizes[i],layer_sizes[i+1]),mid_activation_func()])
        layers[-1] = final_activation_func()

        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)

    def __getitem__(self,key):
        return self.net[key]
        
    


class CQLTrainer:
    def __init__(
        self,
        actor_kwargs:Dict,
        critic_kwargs:List[Dict], # List so that we can have multiple critics.
        sample_data,
    ):
        if isinstance(critic_kwargs,Dict):
            critic_kwargs = [critic_kwargs]

        self.critic_kwargs = critic_kwargs
        self.critics = []
        self.target_critics = []
        for args in critic_kwargs:
            extractor_final = args.get("extractor_final_size",512)
            extractor = CustExtractor(sample_data["obs"],extractor_final)
            layer_sizes = args["hidden_layers"]
            layer_sizes.insert(0,extractor_final + len(sample_data["action"]))
            layer_sizes.append(1)
            net = MLP(layer_sizes,**args)
            critic = Critic(extractor,net)
            self.critics.append(critic)
            self.target_critics.append(copy.deepcopy(critic))

        extractor_final = actor_kwargs.get("extractor_final_size",512)
        actor_extractor = CustExtractor(sample_data["obs"],extractor_final)
        layer_sizes = actor_kwargs["hidden_layers"]
        layer_sizes.insert(0,extractor_final)
        layer_sizes.append(len(sample_data["action"]))
        actor_net = MLP(layer_sizes,**actor_kwargs)
        self.actor = ActorProb(actor_extractor,actor_net)

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

    def _get_policy_actions(self,obs,num_actions,network):
        obs_temp = copy.deepcopy(obs)
        if isinstance(obs,Dict):
            for key in obs:
                obs_temp[key] = rep_consecutive(obs_temp[key],num_actions)
        else:
            obs_temp = rep_consecutive(obs_temp,num_actions)

        new_obs_actions,new_obs_log_pi = network(obs_temp,reparam=True,return_log_prob=True)
        return new_obs_actions, new_obs_log_pi

    def _get_tensor_values(self,obs,actions,network):
        a_s = actions.shape[0]
        o_s = obs["tail"].shape[0]
        num_repeat = a_s//o_s

        obs_temp = copy.deepcopy(obs)
        if isinstance(obs,Dict):
            for key in obs:
                obs_temp[key] = rep_consecutive(obs_temp[key],num_repeat)
        else:
            obs_temp = rep_consecutive(obs_temp,num_repeat)

        preds = network(obs_temp,actions)
        preds = preds.view(o_s,num_repeat,1)
        return preds



    def _train_mini_batch(self,batch:pd.DataFrame):
        
        rewards = torch.tensor(batch["reward"])
        # terminals = torch.tensor(batch["done"])
        terminals = torch.zeros_like(rewards)
        obs = batch["obs"]
        actions = torch.cat(batch["action"],dim=0).reshape(-1,5)
        next_obs = batch["next_obs"]

        new_obs_action,log_pi = self.forward(obs)
        # Policy and alpha loss
        
        # Alpha set to 1 as place holder for entropy stuff. TODO: add entropy stuff.
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

    def train(
        self,
        train_dataset:DataLoader,
        max_epochs:int,
        batch_size:int,
        val_dataset:DataLoader = None,
        steps_per_epoch:int=np.inf, # To have something similar to the repo this code is based off of.
        callback:Callable=lambda *args,**kwargs:None
    ):
        for epoch in range(max_epochs):
            if steps_per_epoch == np.inf:
                for train_batch in train_dataset:
                    self._train_mini_batch(train_batch)
            else:
                for step in range(steps_per_epoch):
                    train_batch = train_dataset.sample(batch_size)
                    self._train_mini_batch(train_batch)


########### Helper Functions
def get_shape(obj):
    if hasattr(obj,"shape"):
        return obj.shape
    
    if isinstance(obj,Dict):
        shapes = [get_shape(val) for val in obj.values()]
            

def rep_consecutive(obj:torch.Tensor,num_rep):
    new_shape = [1] * (len(obj.shape)+1)
    new_shape[1] = num_rep
    return obj.unsqueeze(1).repeat(new_shape).view(obj.shape[0]*num_rep,*obj.shape[1:])