import pandas as pd
from utils.utils_argo import process_data, window_shift
import torch
from torch.utils.data import DataLoader, Dataset
import os 
import numpy as np
import math
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data



def argo_create(path):
    scenes = []
    for fname in os.listdir(path):
        if fname.endswith('csv'):
            scenes.append(path+"/"+fname)
    return scenes




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


class Argo_geometric(InMemoryDataset):
    def __init__(self, obs_len, pred_len, raw_data_path, processed_data_path, split = 'train', 
                root="/home/junanchen/anaconda3/envs/GNN/vehicleTrajectoryForecasting-main/mini/",transform = None,  pre_transform = None):

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.raw_data_path = raw_data_path + split + '/'
        self.processed_data_path = processed_data_path
        super(Argo_geometric, self).__init__(root, transform, pre_transform)

        print('saved to the dir: ',self.processed_paths)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        dirs_in_dir = []
        for r, d, f in os.walk(self.raw_data_path):
            for item in f:
                if '.csv' in item:
                    dirs_in_dir.append(os.path.join(r, item))
        return dirs_in_dir

    @property
    def processed_file_names(self):
        return self.processed_data_path


    def process(self):
        # Read data into huge `Data` list.

        data_list = []

        files = self.raw_file_names

        count = 0
        for file in files:
            count += 1
            print('\r'+str(count)+'/'+str(len(files)),end="")
            # if count>100000:
            #     break

            x, y = process_data(file, self.obs_len, self.pred_len)
            inputs = torch.from_numpy(x).float()

            for i in range(y.shape[0]):
                

            
                label = torch.from_numpy(y[i,:]).float()
                
                data = Data(x=inputs, y=label.unsqueeze(0))
                data_list.append(data)

                inputs = window_shift(inputs, label)


            # print(x)
            # print(y)

            
            num_obj = 5
            edge_index_row2 = torch.linspace(1, num_obj-1, steps=num_obj-1).view(1,-1)
            edge_index_row1 = torch.zeros(num_obj - 1).view(1,-1)
            edge_index = torch.cat((edge_index_row1, edge_index_row2),0)
            edge_index = edge_index.long()





        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    data = Argo(split = 'train')
    dataloader = DataLoader(data, batch_size=3, shuffle = True )

    for i, batch_data in enumerate(dataloader):
        print('batch',batch_data['x'].size())
