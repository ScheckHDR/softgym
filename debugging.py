# from stable_baselines3.common.torch_layers import CustExtractor
# import gym
import numpy as np
import torch as th
from torch import nn

def get_flattened_obs_dim(observation_space) -> int:
    return np.prod(observation_space.shape)

class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()



class CustExtractor(nn.Module):

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__()#observation_space, features_dim)

        self.model = nn.Sequential(
            nn.Tanh(),
            nn.Sigmoid(),
            nn.Identity(),
        )        
    def forward(self,observations: th.Tensor) -> th.Tensor:
        pass
        #return self.model(observations)


class FlattenExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space):
        super(FlattenExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)

points = 3#41
observation_space = np.array([[-1]*points],ndmin=3)
test = th.ones(*observation_space.shape)
a = CustExtractor(observation_space)
b = FlattenExtractor(observation_space)
print('Cust:')
a._slow_forward(test)
print('flatten:')
b(test)