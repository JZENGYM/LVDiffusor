import os
import pickle
import random
import sys
import time
from collections import deque
from weakref import ref

import cv2
#import ebor 
# import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from ipdb import set_trace
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from scipy.spatial import distance_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torchvision.utils import make_grid
from tqdm import tqdm

"""
def split_dataset(dataset, seed, test_ratio, full_train='False'):
    random.seed(seed)

    # get train and test indices
    # get test split according to mode
    if dataset.mode == 'multi':
        items_dict = dataset.items_dict
        test_num = int(len(items_dict.keys()) * test_ratio)
        test_keys = random.sample(list(items_dict.keys()), test_num)
        test_indics = []
        for key in test_keys:
            test_indics += items_dict[key]
    else:
        test_num = int(len(dataset) * test_ratio)
        test_indics = random.sample(range(len(dataset)), test_num)

    # get train according to test
    train_indics = list(set(range(len(dataset))) - set(test_indics))

    # assertion of indices
    assert len(train_indics) + len(test_indics) == len(dataset)
    assert len(set(train_indics) & set(test_indics)) == 0

    # split dataset according to indices
    test_dataset = Subset(dataset, test_indics)
    train_dataset = dataset if full_train == 'True' else Subset(dataset, train_indics)

    # log infos
    infos_dict = {
        'test_indices': test_indics,
        'train_indices': train_indics,
        'room_num': len(dataset.items_dict.keys()),
    }
    return train_dataset, test_dataset, infos_dict

"""
class GraphDataset:
    def __init__(self, data_name, data_root, rotation=True, return_names=True, base_noise_scale=0.01):
        # self.data_root = f'./datasets/{data_name}/'
        # self.folders_path = os.listdir(self.data_root)
        # self.folders_path = f'/../data/{data_name}'
        # self.folders_path = folder_path
        # self.folders_path = f'./datasets/{data_name}/data/'
        self.folders_path = data_root
        self.rotation = rotation
        self.items = []
        self.items_dict = {}
        ptr = 0
        # for files in self.folders_path:
            # print(files)
            # cur_folder_path = f'{self.folders_path}/two_sides/'
            # cur_folder_path = self.folders_path
        files_list = os.listdir(self.folders_path)
        if 'img_bbox' not in self.items_dict.keys():
            self.items_dict['img_bbox'] = []
        for idx in range(len(files_list)):
            item = {
                'obj_path': self.folders_path + files_list[idx],
                'scene_name': data_name
            }
            print(files_list[idx])
            self.items.append(item)
            # self.items_dict['img_bbox'].append(ptr)
            ptr += 1

        if self.rotation==True:
            self.state_dim = 4
        else:
            self.state_dim = 2
            
        print(f'state_dim = {self.state_dim}')
        self.size_dim = 2
        self.scale = base_noise_scale
        self.histogram_path = f'./histogram.png'

        # self.draw_histogram()

        # self.mode = 'multi' if len(self.items_dict.keys()) > 1 else 'single'
        self.mode = 'single'
        self.return_names = return_names

    def draw_histogram(self):
        plt.figure(figsize=(10, 10))
        histogram = []
        for files in self.folders_path:
            cur_folder_path = f'{self.data_root}/{files}/'
            files_list = os.listdir(cur_folder_path)
            histogram.append(len(files_list) // 2)
        histogram = np.array(histogram)
        plt.hist(histogram, bins=4)
        plt.title(f'Total room num: {len(self.folders_path)}')
        plt.savefig(self.histogram_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        item_path = self.items[item]
        with open(item_path['obj_path'], 'rb') as f:
            obj_batch = pickle.load(f)
            obj_batch = np.array(obj_batch)
        
        ''' else, we prepro numpy data into tensors/graphs ''' 
        # wall_feat = torch.tensor(wall_feat).float()

        # # add orientation test
        # ori = np.array([[0, 1]]) # sin, cos
        # obj_batch = np.insert(obj_batch, 2, values=ori, axis=0)
        obj_batch = torch.tensor(obj_batch)
        edge_obj = knn_graph(obj_batch[:, 0:2], obj_batch.shape[0] - 1)  # fully connected
        spatial_state = obj_batch[:, 0:self.state_dim].float()
        spatial_state[:, :2] = spatial_state[:, :2]*2 - 1
        # print(spatial_state)
        if self.rotation:
            data_obj = Data(x=spatial_state,
                            geo=obj_batch[:, self.state_dim:self.state_dim + self.size_dim].float(),
                            category=obj_batch[:, -1:].long(),
                            edge_index=edge_obj)
        else:
            data_obj = Data(x=spatial_state,
                            geo=obj_batch[:, self.state_dim : self.state_dim + self.size_dim].float(),
                            category=obj_batch[:, -1:].long(),
                            edge_index=edge_obj)
        # augment the data with slight perturbation
        scale = self.scale

        data_obj.x += torch.randn_like(data_obj.x) * scale
        if self.return_names:
            return data_obj
            # return data_obj, item_path['room_name']

        return data_obj