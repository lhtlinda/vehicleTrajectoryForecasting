import pandas as pd
from data.utils_argo import compute
import torch
from torch.utils.data import DataLoader, Dataset
import os 
import numpy as np
import math
from numpy import cos, sin


def transform(theta, tracks_obs):
    th = -theta + np.pi/2
    Rot = np.array([[cos(th),-sin(th)],[sin(th),cos(th)]])
    new_pos = (np.dot(Rot, tracks_obs.reshape(-1,2).T)).T
    return new_pos

def argo_create(path):
    scenes = []
    for fname in os.listdir(path):
        if fname.endswith('csv'):
            scenes.append(path+"/"+fname)
    return scenes


def process_data(data_path, obs_len, pred_len):
    # argo_data = argo_create(data_path)

    # count = 0
    # for scene_name in argo_data.keys():
        
    # read data
    tracks_obs, agent_track_pred = compute(data_path, obs_len)

    # create edge_index
    num_obj = tracks_obs.shape[0]
    edge_index_row2 = torch.linspace(1, num_obj-1, steps=num_obj-1).view(1,-1)
    edge_index_row1 = torch.zeros(num_obj - 1).view(1,-1)
    edge_index = torch.cat((edge_index_row1, edge_index_row2),0)
    edge_index = edge_index.long()

    # 
    tracks_obs = tracks_obs[:2,:,3:5].astype('float64')
    agent_track_pred = agent_track_pred[:,3:5].astype('float64')


    # normalize            
    agent_track_pred = agent_track_pred - tracks_obs[0,-1,:]
    tracks_obs = tracks_obs - tracks_obs[0,-1,:]  

    # cal velocity
    v = np.zeros_like(tracks_obs)
    for i in range(tracks_obs.shape[0]):
        for j in range(tracks_obs.shape[1]-1):
            v[i,j,:] = tracks_obs[i,j+1,:] - tracks_obs[i,j,:]
        v[i,-1,:] = v[i,-2,:]
    
    # cal acceleration
    a = np.zeros_like(tracks_obs)
    for i in range(v.shape[0]):
        for j in range(a.shape[1]-1):
            a[i,j,:] = v[i,j+1,:] - v[i,j,:]
        a[i,-1,:] = a[i,-2,:]

    # cal theta
    theta = np.zeros(v.shape[1])
    for k in range(v.shape[1]):
        if v[0,k,1]>0 and v[0,k,0] >0:
            theta[k] = math.atan(v[0,k,1]/v[0,k,0])
        if v[0,k,1]>0 and v[0,k,0] <0:
            theta[k] = np.pi - math.atan(v[0,k,1]/(-v[0,k,0]))
        if v[0,k,1]<0 and v[0,k,0] <0:
            theta[k] = math.atan((-v[0,k,1])/(-v[0,k,0])) + np.pi
        if v[0,k,1]<0 and v[0,k,0] >0:
            theta[k] = -math.atan((-v[0,k,1])/v[0,k,0])

        if theta[k]>2*np.pi:
            theta[k,0] = theta[k] - 2*np.pi
        if theta[k]<0:
            theta[k] = theta[k] + 2*np.pi

        if k>0:
            if theta[k]-theta[k-1]>np.pi:
                theta[k] -= 2*np.pi
            elif theta[k]-theta[k-1]<-np.pi:
                theta[k] += 2*np.pi        
    

    # transform coodinate
    # obs
    th = np.mean(theta[-4:])
    track_obs = transform(th, tracks_obs).reshape(-1,obs_len,2)
    v = transform(th, v).reshape(-1,obs_len,2)
    a = transform(th, a).reshape(-1,obs_len,2)
    new_tracks_obs = np.concatenate((track_obs,v,a),2)
    # pred
    new_agent_pred = transform(th, agent_track_pred).reshape(pred_len,2)


    x = torch.from_numpy(new_tracks_obs).float()
    y = torch.from_numpy(new_agent_pred).float()

    return x, y 

        # count += 1

        # print('\r'+str(count)+'/'+str(len(argo_data)),end="")




class Argo(Dataset):
    def __init__(self, root="/home/junanchen/anaconda3/envs/GNN/vehicleTrajectoryForecasting-main/mini/", split="train", obs_len=20, pred_len=30):
        self.obs_len, self.pred_len = obs_len, pred_len
        self.root = root + split + '/'
        self.argo_data = argo_create(self.root)


    def __getitem__(self, item):
        data_path = self.argo_data[item]
        x, y = process_data(data_path, self.obs_len, self.pred_len)
        sample = {'x':x, 'y':y}
        return sample

    def __len__(self):
        return len(self.argo_data)




if __name__ == '__main__':
    data = Argo(split = 'train')
    dataloader = DataLoader(data, batch_size=3, shuffle = True )

    for i, batch_data in enumerate(dataloader):
        print('batch',batch_data['x'].size())
