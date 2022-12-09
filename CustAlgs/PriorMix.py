import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import cv2
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn,  TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from torch.distributions.normal import Normal


import softgym.utils.topology as topology


class TopologyMix(SAC):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        goal = self.env.get_attr("goal")[0]
        img=self.env.env_method("render_no_gripper")[0]
        topo_state = self.env.env_method("get_topological_representation")[0]

        

        topo_plan = topology.find_topological_path(topo_state,goal,goal.size)
        pick_idxs, pick_region,mid_region,place_region = topology.topo_to_geometry(topo_state,action=topo_plan[1].action)

        pick_mu = pick_idxs[len(pick_idxs)//2] / 40
        pick_std = (len(pick_idxs)//2) / 41

        pick_mid_abs = pick_region[pick_region.shape[0]//2,:]
        mid_region_rel = mid_region - pick_mid_abs
        place_region_rel = place_region - pick_mid_abs

        mid_x_mu, mid_y_mu = np.mean(mid_region_rel,axis=0)
        mid_x_std, mid_y_std = (np.max(mid_region,axis=0) - np.min(mid_region,axis=0)) / 2

        place_x_mu, place_y_mu = np.mean(place_region_rel,axis=0)
        place_x_std, place_y_std = (np.max(place_region,axis=0) - np.min(place_region,axis=0)) / 2

        prior_mus = torch.tensor(np.array([pick_mu,mid_x_mu,mid_y_mu,place_x_mu,place_y_mu])).to("cuda")
        prior_stds = torch.tensor(np.array([pick_std,mid_x_std,mid_y_std,place_x_std,place_y_std])).to("cuda")
        prior_stds = torch.max(prior_stds,torch.ones_like(prior_stds)*0.01)

        policy_mus, policy_stds,_ = self.actor.get_action_dist_params(torch.tensor(observation).to("cuda"))

        combined_mus = (prior_mus*policy_stds**2 + policy_mus*prior_stds**2)/(policy_stds**2 + prior_stds**2)
        combined_stds = torch.sqrt((policy_stds**2 * prior_stds**2)/(prior_stds**2 + policy_stds**2))

        combined_normal = Normal(combined_mus,combined_stds)



        # For visualising.
        x,y,z,theta = self.env.env_method("get_rope_frame")[0]
        T_mat = np.array([
            [np.cos(theta),np.sin(theta),x],
            [-np.sin(theta),np.cos(theta),z],
            [0,0,1]
        ])
        show_prior_on_image(img,pick_region,mid_region,place_region,lambda x: T_mat@x)

        # test = super().predict(observation,state,episode_start,deterministic)
        return combined_normal.rsample().detach().cpu().numpy(),None


        



    def _get_model_dist(self):
        return None




########################## Helper Functions
def show_prior_on_image(img:np.ndarray,pick_region:np.ndarray,mid_region:np.ndarray,place_region:np.ndarray,data_transform = lambda x: x):
    h = img.shape[0]
    w = img.shape[1]
    s = 0.35
    homography,_ = cv2.findHomography(
        np.array([
            [-s, s, s,-s],
            [-s,-s, s, s],
            [ 1, 1, 1, 1],

        ]).T,
        np.array([
            [0,w,w,0],
            [0,0,h,h],
            [1,1,1,1],
        ]).T
    )

    # Ensure regions are column vectors
    if pick_region.shape[0] != 2:
        pick_region = pick_region.T
    assert pick_region.shape[0] == 2
    if mid_region is None:
        mid_region = np.array([0,0],ndmin=2)
    if mid_region.shape[0] != 2:
        mid_region = mid_region.T
    assert mid_region.shape[0] == 2
    if place_region.shape[0] != 2:
        place_region = place_region.T
    assert place_region.shape[0] == 2

    pick_h = np.vstack([pick_region,np.ones(pick_region.shape[1])])
    mid_h = np.vstack([mid_region,np.ones(mid_region.shape[1])])
    place_h = np.vstack([place_region,np.ones(place_region.shape[1])])

    pick_img_p = (homography @ data_transform(pick_h))[:2,:]
    mid_img_p = (homography @ data_transform(mid_h))[:2,:]
    place_img_p = (homography @ data_transform(place_h))[:2,:]



    pick_frame = cv2.polylines(
        np.zeros_like(img),
        [pick_img_p.T.astype(np.int32)],
        isClosed=False,
        thickness=3,
        color=(0,255,0)
    )
    mid_frame = cv2.fillPoly(
        np.zeros_like(img),
        [mid_img_p.T.astype(np.int32)],
        color=(0,0,255)
    )
    place_frame = cv2.fillPoly(
        np.zeros_like(img),
        [place_img_p.T.astype(np.int32)],
        color=(255,0,0)
    )

    info_img = cv2.addWeighted(pick_frame,1,cv2.addWeighted(mid_frame,1,place_frame,1,0),1,0)
    img_final = cv2.addWeighted(info_img,0.5,img,0.5,0)

    # img_p = cv2.bitwise_or(img,pick_frame)
    # img_m = cv2.bitwise_or(img_p,mid_frame)
    # img_final = cv2.bitwise_or(img_m,place_frame)


    cv2.imshow("action_vis",img_final)
    cv2.waitKey(1)