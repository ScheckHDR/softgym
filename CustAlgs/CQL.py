'''
https://github.com/polixir/OfflineRL/blob/3ef71477daa7c165996b00dcdf89716f2b22b8d8/offlinerl/algo/modelfree/cql.py#L102
used to base this file off of.
'''

import copy
import numpy as np
from typing import Callable, List, Union, Dict, Optional
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data.dataloader import DataLoader
from torch import optim

import wandb


class CustExtractor(nn.Module):

    def __init__(self, sample_obs ,out_dim: int = 512,device = 'cuda' if torch.cuda.is_available() else 'cpu',*args,**kwargs):
        super().__init__()
        self.device = device
        self.cnn = nn.Sequential(
            # nn.Identity()
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)
        # layers = [nn.Identity()]

        # Compute shape by doing one forward pass
        with torch.no_grad():
            test = torch.as_tensor(sample_obs['cross']).unsqueeze(0).unsqueeze(0).float().to(device)
            n_flatten = self.cnn(test).shape[1] + np.prod(sample_obs['tail'].shape) + np.prod(sample_obs['shape'].shape)

        self.linear = nn.Sequential(nn.Linear(n_flatten, out_dim), nn.ReLU()).to(device)
        
        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_result = self.cnn(observations['cross'].unsqueeze(1).float().to(self.device))
        # print(observations['cross'].shape)
        # print(f'tail:{observations["tail"].flatten(1).shape}')
        # print(f'shape: {observations["shape"].flatten(1).shape}')
        # print(f'cnn: {cnn_result.shape}')
        lin_input = torch.cat([
            cnn_result,
            observations['tail'].flatten(1).float().to(self.device),
            observations['shape'].flatten(1).float().to(self.device)
            ],dim=1
        )
        return self.linear(lin_input)

    def __get_item__(self,key):
        return (self.cnn[:] + self.linear[:])[key]

class Critic(nn.Module):
    def __init__(
        self,
        state_extractor,
        network:nn.Module,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = device
        self.extractor = state_extractor
        self.Q = network

    def forward(
        self,
        state,
        action,
    ):
        logits = self.extractor(state)

        logits_a = torch.cat([logits,action.to(self.device)],dim=1).float()
        return self.Q(logits_a)

    def __get_item__(self,key):
        return (self.extractor[:] + self.Q[:])[key]

class ActorProb(nn.Module):
    def __init__(
        self,
        state_extractor,
        network:Optional[nn.Module] = None,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = device
        self.extractor = state_extractor
        self.net = network

        self.sigma = nn.Parameter(torch.zeros(1,network[-2].out_features)).to(device)

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
        log_prob = normal.log_prob(actions.to(self.device))
        return log_prob.sum(-1)

    def __getitem__(self,key):
        return (self.extractor[:] + self.network[:])[key]

class MLP(nn.Module):
    def __init__(self,layer_sizes,mid_activation_func=nn.ReLU,final_activation_func=nn.Identity,device = 'cuda' if torch.cuda.is_available() else 'cpu',*args,**kwargs):
        super().__init__()
        assert len(layer_sizes) > 1, f'Number of layers (including input and output) must be at least 2.'

        self.device = device 
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.extend([nn.Linear(layer_sizes[i],layer_sizes[i+1]),mid_activation_func()])
        layers[-1] = final_activation_func()

        self.net = nn.Sequential(*layers).to(device)

    def forward(self,x):
        return self.net(x.to(self.device))

    def __getitem__(self,key):
        return self.net[key]
        
class CQLTrainer:
    def __init__(
        self,
        actor_kwargs:Dict,
        critic_kwargs:Dict,
        sample_data,
        gamma,
        explore = 1, # WTF is explore???
        temperature = 1,
        min_q_weight = 0,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):

        self.critic_criterion = nn.MSELoss()
        self.gamma = gamma
        self.explore = explore
        self.temperature = temperature
        self.min_q_weight = min_q_weight
        self.device = device
        self._n_train_steps_total = 0

        self.critic_kwargs = critic_kwargs


        extractor_final = critic_kwargs.get("extractor_final_size",512)
        extractor = CustExtractor(sample_data["obs"],extractor_final,device=device)
        layer_sizes = critic_kwargs["hidden_layers"]
        layer_sizes.insert(0,extractor_final + len(sample_data["action"]))
        layer_sizes.append(1)
        net = MLP(layer_sizes,device=device,**critic_kwargs)
        critic1 = Critic(extractor,net)
        self.critic1 = critic1
        self.target_critic1 = copy.deepcopy(critic1)
        self.critic_optim1 = optim.Adam(critic1.parameters(),lr=critic_kwargs["lr"])
        self.critic_update_taus = critic_kwargs["tau"]

        extractor_final = actor_kwargs.get("extractor_final_size",512)
        actor_extractor = CustExtractor(sample_data["obs"],extractor_final,device=device)
        layer_sizes = actor_kwargs["hidden_layers"]
        layer_sizes.insert(0,extractor_final)
        layer_sizes.append(len(sample_data["action"]))
        actor_net = MLP(layer_sizes,device=device,**actor_kwargs)
        self.actor = ActorProb(actor_extractor,actor_net)
        self.actor_optim = optim.Adam(self.actor.parameters(),lr=actor_kwargs["lr"])

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
        batch = move_to(batch,self.device)
        
        rewards = batch["reward"].unsqueeze(1)
        # terminals = batch["done"]
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
        if False:
            policy_log_prob = self.actor.log_prob(obs,actions)
            policy_loss = (alpha*log_pi-policy_log_prob).mean()
        else:
            q_new_actions = self.critic1(obs,new_obs_action).detach()
            policy_loss = (alpha*log_pi - q_new_actions).mean()

        # Actor update
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Q-function loss.
        q1_pred = self.critic1(obs,actions)
        new_curr_action, new_curr_log_pi = self.forward(obs,reparam=True,return_log_prob=True)
        new_next_actions, new_log_pi = self.forward(next_obs,reparam=True,return_log_prob=True)

        # Using max q backup. TODO: Other options.
        next_actions_temp,_ = self._get_policy_actions(next_obs,num_actions=10,network=self.forward)
        target_QF_values = self._get_tensor_values(next_obs,next_actions_temp,network=self.critic1).max(1)[0].view(-1,1)
        target_Q_values = torch.min(target_QF_values)

        q_target = rewards + (1-terminals) * self.gamma * target_Q_values.detach()

        q1_loss = self.critic_criterion(q1_pred,q_target)

        # # Add CQL.
        # rand_act = torch.FloatTensor(q1_pred.shape[0]*10,actions.shape[-1]).uniform_(-1,1).to(self.device)
        # curr_act_tens,curr_log_pis = self._get_policy_actions(obs,10,network=self.forward)
        # new_curr_act_tens,new_log_pis = self._get_policy_actions(next_obs,10,network=self.forward)
        # q1_rand = self._get_tensor_values(obs,rand_act,network=self.critic1)
        # q1_curr_actions = self._get_tensor_values(obs,curr_act_tens,network=self.critic1)
        # q1_next_actions = self._get_tensor_values(obs,new_curr_act_tens,network=self.critic1)

        # q1_cat = torch.cat([q1_rand,q1_pred.unsqueeze(1),q1_next_actions,q1_curr_actions],dim=1)

        # # TODO: Something about importance sampling.


        # min_q1_loss = (torch.logsumexp(q1_cat/self.temperature,dim=1).mean()*self.temperature - q1_pred.mean())*self.min_q_weight
        min_q1_loss = 0 # explore is set so that this will get ignored anyway.

        # TODO: Something about Lagrange thresh.


        q1_loss = self.explore*q1_loss + (1-self.explore)*min_q1_loss


        # Update Critics
        self.critic_optim1.zero_grad()
        q1_loss.backward(retain_graph=False)
        self.critic_optim1.step()

        sync_weight(self.target_critic1,self.critic1,self.critic_update_taus)

        self._n_train_steps_total += 1

        return {
            "actor_loss" : policy_loss,
            "critic_loss" : q1_loss,
        }
        
    def get_validation_loss(self,batch):
        with torch.no_grad():
            batch = move_to(batch,self.device)
            
            # rewards = batch["reward"].unsqueeze(1)
            # terminals = batch["done"]
            # terminals = torch.zeros_like(rewards)
            obs = batch["obs"]
            actions = torch.cat(batch["action"],dim=0).reshape(-1,5)
            # next_obs = batch["next_obs"]

            new_obs_action,log_pi = self.forward(obs)

            alpha = 0 # TODO, proper way to do this shit.
            policy_log_prob = self.actor.log_prob(obs,actions)
            policy_loss = (alpha*log_pi-policy_log_prob).mean()

        return policy_loss

    def train(
        self,
        train_dataset:DataLoader,
        max_epochs:int,
        batch_size:int,
        val_dataset:DataLoader = None,
        steps_per_epoch:int=np.inf, # To have something similar to the repo this code is based off of.
        callback:Callable=lambda *args,**kwargs:None
    ):
        for epoch in tqdm(range(max_epochs)):
            callback_data = {
                "losses" : {
                    "actor_loss" : [],
                    "critic_loss" : [],
                    "actor_validation_loss" : [],
                }
            }
            num_batches = 0
            if steps_per_epoch == np.inf:
                for train_batch in train_dataset:
                    losses = self._train_mini_batch(train_batch)
                    for key in losses:
                        callback_data["losses"][key].append(losses[key].cpu().detach())
                    num_batches += 1
            else:
                for step in range(steps_per_epoch):
                    train_batch = train_dataset.sample(batch_size)
                    self._train_mini_batch(train_batch)

            # Validation
            with torch.no_grad():                
                for val_batch in val_dataset:
                    callback_data["losses"]["actor_validation_loss"].append(self.get_validation_loss(val_batch).cpu().detach())
            callback_data["models"] = {
                "actor" : self.actor,
                "critic": self.critic1
            }
            callback(callback_data)

            

        return self.actor,self.critic1


########### Helper Functions
def sync_weight(target:nn.Module,net:nn.Module,tau:float=5e-3):
    for t,n in zip(target.parameters(),net.parameters()):
        t.data.copy_(t.data * (1-tau) + n.data*tau)



def move_to(obj,device):
    if hasattr(obj,'to'):
        obj = obj.to(device)
    elif isinstance(obj,Dict):
        for key in obj:
            obj[key] = move_to(obj[key],device)

    return obj

def get_shape(obj):
    if hasattr(obj,"shape"):
        return obj.shape
    
    if isinstance(obj,Dict):
        shapes = [get_shape(val) for val in obj.values()]
            

def rep_consecutive(obj:torch.Tensor,num_rep):
    new_shape = [1] * (len(obj.shape)+1)
    new_shape[1] = num_rep
    return obj.unsqueeze(1).repeat(new_shape).view(obj.shape[0]*num_rep,*obj.shape[1:])