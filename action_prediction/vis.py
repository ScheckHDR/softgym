import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
import pandas as pd
import time

import torch
from torch.nn.functional import mse_loss
try:
    from action_prediction.TrainPredictors import StatePredictor, ActionPredictor
except:
    from TrainPredictors import StatePredictor, ActionPredictor

def plot(before:tuple,action:tuple,after:tuple,loss:float):
    '''
    each input should be either a tuple of two elements (ground truth and test) as np arrays,
    or a tuple of tuples, where each nested tuple has two elements as before.
    '''

    if type(before[0]) == tuple:
        pass
    else:
        before_g = np.reshape(before[0].detach().numpy(),(5,10))
        before_t = np.reshape(before[1].detach().numpy(),(5,10))

        action_g = action[0].detach().numpy()
        action_t = action[1].detach().numpy()

        after_g = np.reshape(after[0].detach().numpy(),(5,10))
        after_t = np.reshape(after[1].detach().numpy(),(5,10))


        plt.plot(before_g[0,:],before_g[1,:],'g-')
        # plt.plot(before_t[0,:],before_t[1,:],'r-')

        plt.plot(after_g[0,:],after_g[1,:],'b--')
        plt.plot(after_t[0,:],after_t[1,:],'r--')

        plt.text(-0.3,0.3,f'loss:{loss}')

        # plt.show(block=block)


def get_ground_truth_sample(dataset):

    rand_idx = np.random.randint(0,len(dataset))
    print(rand_idx)
    before, action, after = dataset.iloc[rand_idx,:]
        

    before = np.fromstring(before.strip('[]'),np.float32,50,' ')
    action = np.fromstring(action.strip('[]'),np.float32,3,' ')
    after  = np.fromstring(after.strip('[]'),np.float32,50,' ')
    # print(before)
    # print(action)
    # print(after)
    # print('-'*50)

    return torch.tensor(before),torch.tensor(action),torch.tensor(after)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path',type=str)
    parser.add_argument('dataset_path',type=str)
    parser.add_argument('--t',type=float,default=0,help='The amount of time to leave visualisation on screen before moving to next. If 0, manual mode.')
    parser.add_argument('--mode',type=str,default='FORWARD')

    args = parser.parse_args()

    args.mode = args.mode.upper()

    assert exists(args.model_path), f'Model file {args.model_path} does not exist.'
    assert exists(args.dataset_path), f'Dataset file {args.dataset_path} does not exist.'
    assert args.t >= 0, f't must be a non-negative real-valued number, not {args.t}'
    assert args.mode in ['FORWARD','BACKWARD'], f'Mode must be either FORWARD or BACKWARD, not {args.mode}'

    return args



if __name__ == '__main__':
    args = get_args()

    dataset = pd.read_csv(args.dataset_path,sep=',')
    
    # model = Predictor(args.sizes)
    if args.mode == 'FORWARD':
        Predictor = StatePredictor
    elif args.mode == 'BACKWARD':
        Predictor = ActionPredictor
    model = Predictor.load_from_checkpoint(args.model_path)
    # model = torch.load(args.model_path)
    # print(model)
    model.eval()


    plt.xlim(-0.5,0.5), plt.ylim(-0.5,0.5)
    plt.ion()
    plt.show()

    while True:
        before_g,action_g,after_g = get_ground_truth_sample(dataset)
        

        if args.mode == 'FORWARD':
            model_input = np.concatenate([deepcopy(before_g),deepcopy(action_g)])
            before_t,action_t,after_t = deepcopy(before_g),deepcopy(action_g),model(torch.tensor(model_input))
            loss = mse_loss(after_t,after_g)
        elif args.mode == 'BACKWARD':
            model_input = np.concatenate([deepcopy(before_g),deepcopy(after_g)])
            before_t,action_t,after_t = deepcopy(before_g),model(torch.tensor(model_input)),deepcopy(after_g)
            loss = mse_loss(action_t,action_g)
        else:
            raise NotImplementedError

        plot((before_g,before_t),(action_g,action_t),(after_g,after_t),loss)
        plt.draw()
        plt.xlim(-0.3,0.3), plt.ylim(-0.3,0.3)
        if args.t == 0:
            input()
        else:
            plt.pause(args.t)
        plt.clf()

