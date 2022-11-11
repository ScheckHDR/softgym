import argparse
import numpy as np
import os
import pandas as pd
import pickle
from softgym.utils.topology import get_topological_representation
from tqdm import tqdm
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from CustAlgs.CQL import CQLTrainer

import wandb

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
        val = abs(obj["losses"]["actor_validation_loss"])
        if val < best_val:
            best_val = val
            torch.save(obj["models"],f"{path}/data2_batch_size{wandb.config.batch_size}_alr{wandb.config.actor_lr}_clr{wandb.config.critic_lr}_gamma{wandb.config.gamma}.pth")

    return foo
class RopeDataset(Dataset):
    def __init__(self,data_file,goal_topology = None,size=None,action_types=None):
        with open(data_file,'rb') as f:
            self.df = pd.DataFrame(pickle.load(f))

        self.df.insert(0,"action_type",[action[0] for action in self.df["action"]])
        self.df["action"] = [action[1:] for action in self.df["action"]]
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

def topology_from_obs(obs):
    incidence = obs['cross']
    pos = np.hstack([np.zeros((2,1)),obs['shape']]).T
    pos = np.insert(pos,1,np.sum(incidence,axis=1),axis=1)

    return get_topological_representation(pos)

    


def recompute_rewards(dataset,goal_topology):
    print('Recomputing Rewards for goal topology.')
    for i in tqdm(range(len(dataset))):
        t = topology_from_obs(dataset.iloc[i]['next_obs'])
        if np.all(t == goal_topology):
            dataset.iloc[i]['reward'] = 0
        else:
            dataset.iloc[i]['reward'] = -1

    return dataset




    


if __name__ == '__main__':
    goal_topology = np.array([
        [0,1],
        [1,0],
        [1,-1],
        [1,1]
    ])


    default_config = {
        "batch_size" : 32,
        "actor_lr" : 1e-3,
        "critic_lr" : 1e-3,
        "gamma" : 0.95,
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
        goal_topology,
        # size=1000,
        action_types=['+C']
    )
    actor_args = {
        "hidden_layers" : [128,128],
        "lr" : wandb.config.actor_lr,
        "extractor_final_size" : 512,
    }
    critic_args = {
        "hidden_layers" : [128,128],
        "lr" : wandb.config.critic_lr,
        "tau" : wandb.config.tau,
        "extractor_final_size" : 512,
    }

    sample_data = data[0]
    trainer = CQLTrainer(
        actor_args,
        critic_args,
        sample_data,
        gamma = 0.9,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset,val_dataset = torch.utils.data.random_split(data,[train_size,val_size])

    cb = callback_list([
        log_callback,
        save_callback()
    ])

    actor,critics = trainer.train(
        DataLoader(train_dataset,wandb.config.batch_size,shuffle=True),
        wandb.config.max_epochs,
        wandb.config.batch_size,
        val_dataset=DataLoader(val_dataset,wandb.config.batch_size,shuffle=True),
        callback=cb
    )

    del actor
    del critics
    run.finish()    



    
    # torch.save(actor,'./actor.model')
    # for i in range(len(critics)):
    #     torch.save(critics[i],f'./critic{i}.model')