import argparse
import numpy as np
from os.path import exists
from typing import Dict

import torch

from softgym.envs.rope_knot import RopeKnotEnv
from softgym.utils.normalized_env import normalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def eval_model(
    policy,
    goal_topo,
    num_episodes:int = 100, 
    max_episode_length:int = 5,
    num_workers:int = 1,
    headless:int = True,
):
    assert num_episodes % num_workers == 0, f"For simplicity, num_episodes must be an integer multiple of num_workers."
    env_kwargs = {
        "observation_mode": "key_point",
        "action_mode": "picker_trajectory",
        "num_picker": 1,
        "render": not headless,
        "headless": headless,
        "horizon": max_episode_length + 1, # Addd one because I think the env gets marked as done if it reaches the limit. I want to test that separately.
        "action_repeat": 1,
        "render_mode": "cloth",
        "num_variations": 500,
        "use_cached_states": True,
        "save_cached_states": False,
        "deterministic": False,
        "maximum_crossings": 5,
        "goal_crossings": 3,
    }
    
    envs = SubprocVecEnv([lambda: Monitor(RopeKnotEnv(goal_topology=goal_topo,**env_kwargs))]*num_workers,"spawn")
    # envs = Monitor(RopeKnotEnv(goal_topology=goal_topo,**env_kwargs))
    successes = 0
    steps_taken = []
    envs_finished = 0
    
    steps_taken = np.zeros(num_workers)
    with torch.no_grad():
        obs = envs.reset()
        while envs_finished < num_episodes:
           
            obs = as_tensor(obs["shape"].astype(np.float32)).to('cuda')
            # for key in obs:
            #     obs[key] = obs[key].unsqueeze(0)
            action = policy(obs).normal.loc.cpu().numpy()
            obs,rewards,dones,infos = envs.step(action)

            steps_taken += 1
            successes += sum(dones == True)
            dones[dones >= max_episode_length] = True

            reset_idxs = [i for i, done in enumerate(dones) if done]
            reset_obs = envs.env_method('reset',indices = reset_idxs)
            for i in reset_idxs:
                obs[i] = reset_obs[i]
            steps_taken[reset_idxs] = 0

    return successes/num_episodes


def as_tensor(obj):
    if isinstance(obj,Dict):
        for key in obj:
            obj[key] = as_tensor(obj[key])
    else:
        obj = torch.tensor(obj)
    return obj

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path",type=str)

    args = parser.parse_args()

    assert(exists(args.model_path)), f"Model could not be found at {args.model_path}."

    return args

if __name__ == "__main__":
    args = get_args()

    policy = torch.load(args.model_path)["actor"]
    goal_topo = np.array([
        [0,1],
        [1,0],
        [1,-1],
        [1,1]
    ])
    eval_model(policy,goal_topo,num_episodes=10, headless = False)

