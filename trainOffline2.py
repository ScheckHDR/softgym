import numpy as np
import pandas as pd
import pickle

from offlinerl.algo.modelfree.cql import algo_init, AlgoTrainer
import torch
from torch.utils.data import DataLoader,Dataset
import wandb

from typing import List


class RopeDataset(Dataset):
    def __init__(self,data_file,goal_topology = None,size=None,action_types=None):
        with open(data_file,'rb') as f:
            df = pd.DataFrame(pickle.load(f))
        self.df = pd.DataFrame({
            "rew":df["reward"],
            "done":df["dones"],
            "obs":[obs["shape"] for obs in df["obs"]],
            "act":[action[1:] for action in df["action"]],
            "obs_next":[obs["shape"] for obs in df["next_obs"]],
            "action_type": [action[0] for action in df["action"]]
        })

        if action_types is not None:
            self.df = self.df.loc[[a_t in action_types for a_t in self.df["action_type"]]]
            for i in range(len(action_types)):
                self.df.loc[self.df["action_type"] == action_types[i],"action_type"] = i
        if size is not None:
            self.df = self.df.iloc[:size]
        if goal_topology is not None:
            self.df = recompute_rewards(self.df,goal_topology)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.df.iloc[idx]
        return data.to_dict()




def callback_list(callbacks):
    if not isinstance(callbacks,List):
        callbacks = [callbacks]

    def foo(obj):
        for c in callbacks:
            c(obj)

    return foo


def log_callback(obj):
    if "losses" in obj:
        for key in obj["losses"]:
            obj["losses"][key] = np.array(obj["losses"][key]).mean()
        wandb.log(obj["losses"])

def save_callback(path="./TEMP_SWEEP"):
    best_val = np.inf
    def foo(obj):
        nonlocal best_val
        val = abs(obj["losses"]["actor_loss"])
        if val < best_val:
            best_val = val
            torch.save(obj["models"],f"{path}/offlineRL_batch_size{wandb.config.batch_size}_alr{wandb.config.actor_lr}_clr{wandb.config.critic_lr}_gamma{wandb.config.gamma}.pth")

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
        "steps_per_epoch" : 1000,
        "batch_size" : 32,

        "dataset_path": "./Datasets/dataset2.pkl"
    }
    run = wandb.init(
        project="TODO",
        config=default_config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False
    )

    data = RopeDataset(
        wandb.config.dataset_path,
        # goal_topology,
        # size=1000,
        action_types=['+C']
    )


    sample_data = data[0]


    # train_size = int(0.8 * len(data))
    # val_size = len(data) - train_size
    # train_dataset,val_dataset = torch.utils.data.random_split(data,[train_size,val_size])

    cb = callback_list([
        log_callback,
        save_callback()
    ])


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
    soft_target_tau - target network updata, default is 5e-6 I think.

    max_epoch
    steps_per_epoch
    batch_size
    '''
    run_args = {
        "seed" : 1,
        "action_shape" : (1,5),
        "obs_shape" : (2,40),
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

    alg_init = algo_init(run_args)
    trainer = AlgoTrainer(alg_init,run_args)

    trainer.train(data.df,None,cb)


    run.finish()    
