import argparse
import numpy as np
import os
import pandas as pd
import pickle
from softgym.utils.topology import get_topological_representation
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader, Dataset
from CustAlgs.CQL import CQL


class RopeDataset(Dataset):
    def __init__(self,data_file,goal_topology = None,size=None):
        with open(data_file,'rb') as f:
            self.df = pd.DataFrame(pickle.load(f))
        if size is not None:
            self.df = self.df.loc[:size+1]
        if goal_topology is not None:
            self.df = recompute_rewards(self.df,goal_topology)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.df.iloc[idx]
        return data

def topology_from_obs(obs):
    incidence = obs['cross']
    pos = np.hstack([np.zeros((2,1)),obs['shape']]).T
    pos = np.insert(pos,1,np.sum(incidence,axis=1),axis=1)

    return get_topological_representation(pos)



def train(dataset):
    pass


def recompute_rewards(dataset,goal_topology):
    print('Recomputing Rewards for goal topology.')
    for i in tqdm(range(len(dataset))):
        t = topology_from_obs(dataset.loc[i,'next_obs'])
        if np.all(t == goal_topology):
            dataset.loc[i,'reward'] = 0
        else:
            dataset.loc[i,'reward'] = -1

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

    data = RopeDataset(args.dataset_path,goal_topology,1000)
    # print(sum(df['reward'] == 0))
    model = CQL(data)
    model.train()