from typing import Any, Dict, List, Optional, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20



class CustExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        print('init')

        n_input_channels = observation_space['cross'].shape[0]
        self.cnn = nn.Sequential(
            # nn.Identity()
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # layers = [nn.Identity()]

        # Compute shape by doing one forward pass
        with th.no_grad():
            test = th.as_tensor(observation_space['cross'].sample()).unsqueeze(0).float()
            n_flatten = self.cnn(test).shape[1] + np.prod(observation_space['tail'].shape) + np.prod(observation_space['shape'].shape)
            # print('test')

        # input = test    
        # for func in layers:
        #     input = func(input)
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
        

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn_result = self.cnn(observations['cross'])
        # print(observations['cross'].shape)
        # print(f'tail:{observations["tail"].flatten(1).shape}')
        # print(f'shape: {observations["shape"].flatten(1).shape}')
        # print(f'cnn: {cnn_result.shape}')
        lin_input = th.concat([cnn_result,observations['tail'].flatten(1),observations['shape'].flatten(1)],dim=1)
        return self.linear(lin_input)



class CustPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


register_policy('CustPolicy',CustPolicy)