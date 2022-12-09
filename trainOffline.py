import random
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

from offlinerl.algo.modelfree.cql import algo_init, AlgoTrainer
import torch
from torch.utils.data import DataLoader,Dataset
import wandb

from typing import List
from softgym.envs.rope_knot import RopeKnotEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

import softgym.utils.topology as topology

class RopeDataset(Dataset):
    def __init__(self,data_file):
        d = []
        with open(data_file,"rb") as f:
            while True:
                try:
                    d.append(pickle.load(f))
                except EOFError:
                    break
        data = {}
        for key in d[0]:
            data[key] = []
        for i in d:
            for key,val in i.items():
                data[key].extend(val)
        


        self.df = pd.DataFrame({
            "rew": data["rews"],
            "done": data["rews"],
            "obs": [np.concatenate(a,axis=1) for a in zip(data["obs"],np.expand_dims([d[:-1] for d in data["topo_actions"]],1))],
            "act":data["actions"],
            "obs_next":[np.concatenate(a,axis=1) for a in zip(data["obs_next"],np.zeros_like(np.expand_dims([d[:-1] for d in data["topo_actions"]],1)))],
        })

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.df.iloc[idx]
        return data.to_dict()


def all_add_C(topo:topology.RopeTopology) -> List[topology.RopeTopologyAction]:
    added = []
    for over_seg in range(topo.size+1):
        for under_seg in range(topo.size+1):
            if over_seg == under_seg:
                continue
            for chirality in [-1,1]:
                if over_seg in [0,topo.size] or under_seg in [0,topo.size]:
                    action = topology.RopeTopologyAction("+C",over_seg,chirality,under_seg)
                    try:
                        new_topo, _ = topo.add_C(action) # just done to ensure action is valid.
                        added.append(action)
                    except topology.InvalidTopology:
                        pass
    return added
def all_add_R1(topo:topology.RopeTopology) -> List[topology.RopeTopologyAction]:
    added = []
    for seg in range(topo.size+1):
        for chirality in [-1,1]:
            for starts_over in [True,False]:
                action = topology.RopeTopologyAction("+R1",seg,chirality,starts_over=starts_over)
                try:
                    new_topo,_ = topo.add_R1(action)
                    added.append(action)
                except topology.InvalidTopology:
                    pass
    return added

def all_add_actions(topo:topology.RopeTopology) -> List[topology.RopeTopologyAction]:
    added = []
    added.extend(all_add_C(topo))
    added.extend(all_add_R1(topo))
    return added

def callback_list(callbacks):
    if not isinstance(callbacks,List):
        callbacks = [callbacks]

    def foo(obj):
        for c in callbacks:
            obj = c(obj) or obj
    return foo


def log_callback(obj):
    if "logs" in obj:
        for key in obj["logs"]:
            obj["logs"][key] = np.array(obj["logs"][key]).mean()
        wandb.log(obj["logs"])

def save_callback(path="./wandb_sweeps/TEMP_SWEEP"):
    best_val = -np.inf
    os.makedirs(path,exist_ok=True)
    def foo(obj):
        nonlocal best_val
        if "val_return" in obj["logs"]:
            val = abs(obj["logs"]["val_return"])
            if val > best_val:
                best_val = val
                torch.save(obj["models"],f"{path}/offlineRL_batch_size{wandb.config.batch_size}_alr{wandb.config.actor_lr}_clr{wandb.config.critic_lr}_gamma{wandb.config.gamma}.pth")

    return foo

def validation_callback(envs:SubprocVecEnv,val_freq:int=1,num_runs_per_env:int=1):
    calls = 0

    def foo(obj):
        nonlocal calls
        nonlocal envs
        calls += 1
        if calls % val_freq == 0:
            print("Validating.")
            with torch.no_grad():
                actor = obj["models"]["actor"]
                std = []
                returns = []
                obj["logs"]["success_rate"] = 0
                for _ in tqdm(range(num_runs_per_env)):
                    obs = envs.reset()
                    ep_return = np.zeros(envs.num_envs)
                    successes = np.array([False]*envs.num_envs)

                    reps = envs.env_method("get_topological_representation")
                    topo_actions = []
                    for worker_num in range(envs.num_envs):
                        topo_actions.append(random.choice(all_add_actions(reps[worker_num])))
                        envs.env_method("assign_goal",reps[worker_num].take_action(topo_actions[worker_num])[0])

                    for _ in range(envs.get_attr("horizon")[0] - 1):
                        obs_with_topo = [np.concatenate(a,axis=1) for a in zip(obs,np.expand_dims([a.as_array for a in topo_actions],1))]
                        action_dists = actor(torch.tensor(obs_with_topo).float().to("cuda"))
                        actions = action_dists.mode.cpu().numpy()
                        std.append(np.mean(action_dists.normal_std.cpu().numpy(),axis=1))
                        obs,rews,dones,infos = envs.step(actions)
                        successes = np.bitwise_or(successes,dones)
                        ep_return += np.array(rews)
                    returns.append(ep_return)
                    obj["logs"]["success_rate"] += np.sum(successes)
            obj["logs"]["success_rate"] /= num_runs_per_env*envs.num_envs
            obj["logs"]["val_return"] = np.mean(np.array(returns))
            obj["logs"]["action_std_dev"] = np.mean(np.array(std))
        return obj


    return foo


if __name__ == '__main__':



    default_config = {
        "num_layers" : 1,
        "hidden_layer_size" : 128,
        "actor_lr" : 1e-3,
        "critic_lr" : 1e-3,

        "gamma" : 0.95,
        "min_q_weight" : 0.5,
        "explore" : 0,
        "tau" : 5e-6,

        "max_epochs" : 100,
        "steps_per_epoch" : 50,
        "batch_size" : 32,

        "goal" : 0,

        "dataset_path": "Datasets/Prior/Data.pkl"

    }
    run = wandb.init(
        project="Softgym_straighten",
        config=default_config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False
    )

    data = RopeDataset(
        wandb.config.dataset_path,
    )


    sample_data = data[0]

    


    '''
    seed
    action_shape
    obs_shape
    task - leave out
    layer_num
    hidden_layer_size
    actor_lr
    critic_lr
    device
    use_automatic_entropy_tuning
    target_entropy - used if above is set to true
    lagrange_thresh - 0 to disable it appears.

    discrete
    policy_bc_steps
    type_q_backup - {"max","medium","min"}, or anything else to use default.
    q_backup_lmbda - used if previous is set to "medium"
    reward_scale
    discount
    num_random - ???
    min_q_version - ??? < 3 seems to disable whatever this is for
    temp - temperature I'm guessing.
    min_q_weight - ???
    explore - WHY THE FUCK IS THERE A TWO????
    soft_target_tau - target network update, default is 5e-6 I think.

    max_epoch
    steps_per_epoch
    batch_size
    '''
    run_args = {
        "seed" : 1,
        "action_shape" : sample_data["act"].shape,
        "obs_shape" : sample_data["obs"].shape,
        # "task" : "SOME BUILT IN,"
        "layer_num" : wandb.config.num_layers,
        "hidden_layer_size" : wandb.config.hidden_layer_size,
        "actor_lr" : wandb.config.actor_lr,
        "critic_lr" : wandb.config.critic_lr,
        "device" : "cuda" if torch.cuda.is_available() else 'cpu',
        "use_automatic_entropy_tuning" : False,
        # "target_entropy" : 0,
        "lagrange_thresh" : 0,

        "discrete" : False,
        "policy_bc_steps" : 10000,
        "type_q_backup" : "DEFAULT",
        "q_backup_lmbda" : -np.inf,
        "reward_scale" : 1,
        "discount" : wandb.config.gamma,
        "num_random" : 10,
        "min_q_version" : 0, 
        "temp" : 1,
        "min_q_weight" : wandb.config.min_q_weight,
        "explore" : wandb.config.explore,
        "soft_target_tau" : wandb.config.tau,

        "max_epoch" : wandb.config.max_epochs,
        "steps_per_epoch" : wandb.config.steps_per_epoch,
        "batch_size" : wandb.config.batch_size,

        "dataset_path" : wandb.config.dataset_path
    }

    env_kwargs = {
        "observation_mode"  : "key_point",
        "action_mode"       : "picker_trajectory",
        "num_picker"        : 1,
        "render"            : True,
        "headless"          : True,
        "horizon"           : 5 + 1,
        "action_repeat"     : 1,
        # "render_mode"       : args.render_mode,
        "num_variations"    : 500,
        "use_cached_states" : True,#True,
        "save_cached_states": False,
        "deterministic"     : False,
        # "trajectory_funcs"  : [box_trajectory],
        # "maximum_crossings" : args.maximum_crossings,
        "goal" : topology.COMMON_KNOTS[wandb.config.goal],#wandb.config.goal,
        "goal_crossings"    : 3,
        "task" : "KNOT"
    }
    envs = SubprocVecEnv([lambda: RopeKnotEnv(**env_kwargs)]*5,"spawn")
    # envs = RopeKnotEnv(**env_kwargs)

    cb = callback_list([
        validation_callback(envs,10,20),
        log_callback,
        save_callback("./wandb_sweeps/Knot")
    ])

    alg_init = algo_init(run_args)
    trainer = AlgoTrainer(alg_init,run_args)

    trainer.train(data.df,None,cb)


    run.finish()    
