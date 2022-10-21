'''https://github.com/polixir/OfflineRL/blob/master/offlinerl/algo/modelfree/cql.py used as a base to create this.'''

import copy
import pandas as pd

from stable_baselines3.sac.sac import SAC
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import optim


class Network(nn.Module):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -5
    MEAN_MIN = -9.0
    MEAN_MAX = 9.0

    def __init__(self,feature_extractor,input_size,hidden_layer_sizes,output_size,device,activation_func):
        self.feature_extractor = feature_extractor



        if len(hidden_layer_sizes) == 0:
            # Doesn't change the network, prevents code from crashing.
            hidden_layer_sizes = [input_size]

        layers = [nn.Linear(input_size,hidden_layer_sizes[0],device=device),nn.ReLU()]
        for i in range(1,len(hidden_layer_sizes)):
            layers.extend([nn.Linear(hidden_layer_sizes[i-1],hidden_layer_sizes[i],device = device),nn.ReLU()])
        layers.extend([nn.Linear(hidden_layer_sizes[-1],output_size,device=device),activation_func()])

        self.net = nn.Sequential(layers)
        self.sigma = self.sigma = nn.Parameter(torch.zeros(output_size))

    def forward(self,x):
        features = self.feature_extractor(x)
        logits = self.net(features)
        return logits

    def log_prob(self,obs,actions):
        features = self.feature_extractor(obs) 
        output = self.net(features)

        mean = torch.mean(output)
        mean=torch.clamp(mean,self.MEAN_MIN,self.MEAN_MAX)

        shape = [1]*len(mean.shape)
        shape[1] = -1
        log_std = (self.sigma.view(shape)+torch.zeros_like(mean))
        std = log_std.exp()
        
        tanh_normal = TanhNormal(mean,std)



class CQL:
    def __init__(
        self,
        dataset:Dataset,
        feature_extractor,
        feature_extractor_args,
        policy: Union[str, Type[SACPolicy]],
        actor_args:Dict,
        critic_args:Dict,
        learning_rate: Union[float, Schedule] = 3e-4,
        learning_starts: int = 100,
        batch_size: int = 256,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)

        # Init important variables, extract others from args.
        self._current_epoch = 0
        self.num_critics = critic_args.get("num_critics",1)
        assert self.num_critis > 0 and isinstance(self.num_critics,int), f'Must have a positive integer number of critics.'
        self.BC_steps = actor_args.get('BC_steps',0)


        # Initialise models.
        self.critics = []
        self.critic_targets = []
        self.critic_optims = []
        for i in self.num_critics:
            fe = feature_extractor(**feature_extractor_args)
            critic = Network(fe,)
            self.critics.append(critic)
            self.critic_targets.append(copy.deepcopy(critic))
            self.critic_optims.append(optim.Adam(critic.parameters(),lr=critic_args['lr']))
        
        actor_fe = feature_extractor(**feature_extractor_args)
        self.actor = Network(actor_fe,)
        self.actor_optim = optim.Adam(self.actor.parameters(),lr=actor_args['lr'])


    def _train_batch(self,batch:pd.DataFrame):
        obs = batch.obs
        rewards = batch.rewards
        #terminals = batch.done
        actions = batch.action
        next_obs = batch.next_obs


        # Policy loss
        if self._current_epoch < self.BC_steps:
            # Initialise actor by using Behavioural Cloning.
            policy_log_prob = self.actor.log_prob(obs,actions)




    def train(self):
        pass
