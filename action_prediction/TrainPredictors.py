import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint




class StatePredictor(pl.LightningModule):
    def __init__(self,**kwargs):#sizes,lr:float=1e-3,activation=nn.Tanh()):
        super().__init__()

        self.lr = kwargs.get('lr',1e-3)
        activation = kwargs.get('activation',nn.Tanh())

        sizes = [53,100,200,100,50]

        layers = []
        for i in range(1,len(sizes)):
            layers.append(nn.Linear(sizes[i-1],sizes[i]))
            layers.append(activation)
        
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat,y)
        self.log("test_loss",test_loss)
        return test_loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat,y)
        self.log("val_loss",val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = self.lr)

class ActionPredictor(pl.LightningModule):
    def __init__(self,**kwargs):#sizes,lr:float=1e-3,activation=nn.Tanh()):
        super().__init__()

        self.lr = kwargs.get('lr',1e-3)
        activation = kwargs.get('activation',nn.Tanh())

        sizes = [100,100,200,100,3]

        layers = []
        for i in range(1,len(sizes)):
            layers.append(nn.Linear(sizes[i-1],sizes[i]))
            layers.append(activation)
        
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat,y)
        self.log("test_loss",test_loss)
        return test_loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat,y)
        self.log("val_loss",val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = self.lr)

class RopeDataset(Dataset):
    def __init__(self,mode:str,csv_path:str,transform=None):
        mode = mode.upper()

        assert mode in ['FORWARD','BACKWARD'], f'Only FORWARD and BACKWARD modes are accepted.'

        self.mode = mode
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self,idx):
        before, action, after = self.data.iloc[idx,:]
        

        before = np.fromstring(before.strip('[]'),np.float32,50,' ')
        action = np.fromstring(action.strip('[]'),np.float32,3,' ')
        after  = np.fromstring(after.strip('[]'),np.float32,50,' ')




        if self.mode == 'FORWARD':
            x,y = np.concatenate((before,action)),after
        else:
            x,y = np.concatenate((before,after)),action
        #TODO: something with self.transform.

        return x,y



def main(mode:str,args):
    early_stopping_callback = EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.0001,
        patience = 5,
        verbose = True,
        mode = 'min'
    )
    
    if mode == 'forward':
        predictor = StatePredictor(lr=args.lr)
    else:
        predictor = ActionPredictor(lr=args.lr)
    
    dataset = RopeDataset(mode,args.dataset_path)
    train_size = int(len(dataset)*0.8)
    val_size = len(dataset)-train_size
    state_train, state_val = random_split(
        dataset,
        [train_size,val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    checkpoint_callback = ModelCheckpoint(dirpath='./action_prediction/'+mode+'/')
    trainer = pl.Trainer(gpus=1,callbacks=[early_stopping_callback,checkpoint_callback])
    trainer.fit(
        predictor,
        DataLoader(state_train,shuffle=True,num_workers=4,batch_size=args.batch_size),
        DataLoader(state_val,shuffle=False,num_workers=4,batch_size=args.batch_size),   
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path',type=str,help='Path to dataset')
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--batch_size',type=int,default=32)

    args = parser.parse_args()


    return args


if __name__ == '__main__':

    args = get_args()

    main('forward',args)
    main('backward',args)



