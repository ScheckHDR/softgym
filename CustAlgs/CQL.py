from stable_baselines3.sac.sac import SAC
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F




class CQL(SAC):
    def __init__(
        self,
        dataset:Dataset,
        policy: Union[str, Type[SACPolicy]],
        learning_rate: Union[float, Schedule] = 3e-4,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)

    def train(self):
        self.policy.train()

        optimizers = [self.actor.optimizer,self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        

        while True:
            # update learning rate
            self._update_learning_rate(optimizers)

            ent_coef_losses, ent_coefs = [],[]
            actor_losses, critic_losses = [],[]

            for batch in self.dataloader:
                observations = batch['obs']

                # Action by current actor
                actions_pi, log_prob = self.actor.action_log_prob(observations)
                log_prob = log_prob.rehsape(-1,1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None:
                    # Important: detach the variable from the graph
                    # so we don't change it with other losses
                    # see https://github.com/rail-berkeley/softlearning/issues/60
                    ent_coef = torch.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor
                ent_coefs.append(ent_coef.item())

                if ent_coef_loss is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                with torch.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.actor.action_log_prob(batch['next_obs'])
                    # Compute the next Q values: min over all critics targets
                    next_q_values = torch.cat(self.critic_target(batch['next_obs'], next_actions), dim=1)
                    next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    target_q_values = batch['reward'] + (1 - (batch['reward']==0)) * self.gamma * next_q_values

                current_q_values = self.critic(batch['obs'], batch['action'])

                # Compute critic loss
                critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
                critic_losses.append(critic_loss.item())

                # Optimize the critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # Compute actor loss
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Mean over all critic networks
                q_values_pi = torch.cat(self.critic.forward(batch['obs'], actions_pi), dim=1)
                min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()