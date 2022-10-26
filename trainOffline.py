import argparse
import numpy as np
import os
import pandas as pd
import pickle
from softgym.utils.topology import get_topological_representation
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from CustAlgs.CQL import CQLTrainer




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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_path',type=str)


    args = parser.parse_args()

    assert os.path.exists(args.dataset_path), f'Could not find dataset at {args.dataset_path}.'

    return args



if __name__ == '__main__':
    args = get_args()

    

    
    goal_topology = np.array([
        [0,1],
        [1,0],
        [1,-1],
        [1,1]
    ])

    data = RopeDataset(args.dataset_path,goal_topology,1000,['+C'])
    # print(sum(df['reward'] == 0))

    # train(
    #     data,
    #     # seed=args.seed,
    #     # obs_shape=data.df.obs[0].shape,
    #     # action_shape=data.df.action[0].shape,
    #     # layer_num=2,
    #     # hidden_layer_size=128,
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    #     actor_lr=1e-3,
    #     critic_lr=1e-3,
    #     # use_automatic_entropy_tuning=False,
    #     # target_entropy=1e-3,# Might only be used if above is True
    #     # lagrange_thresh=-1, #Might disable it.
    #     # discrete=False,
    #     # policy_bs_steps=0,
    #     # type_q_backup=None, # {max,min,medium}
    #     # reward_scale=1,
    #     gamma=0.9,
    #     # num_random=32, # ???
    #     # min_q_version=None, #???
    #     # temp=0.1,#???
    #     # min_q_weight=1, #???
    #     # explore=0,#???
    #     soft_target_tau=5e-6,#???
    #     max_epoch=100,
    #     # steps_per_epoch=5,#???
    #     batch_size=64
    # )

    actor_args = {
        "hidden_layers" : [128,128],
        "device" : "cuda" if torch.cuda.is_available() else "cpu",
        "gamma" : 0.9,
        "lr" : 1e-3,
        "extractor_final_size" : 512,
    }
    critic_args = [{
        "hidden_layers" : [128,128],
        "device" : "cuda" if torch.cuda.is_available() else "cpu",
        "gamma" : 0.9,
        "lr" : 1e-3,
        "soft_target_tau" : 5e-6,
        "extractor_final_size" : 512,
    }]*2

    sample_data = data[0]
    trainer = CQLTrainer(actor_args,critic_args,sample_data)
    trainer.train(
        DataLoader(data,32,shuffle=True),
        100,
        32
    )