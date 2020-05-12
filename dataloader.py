import os
import glob
import numpy as np

from PIL import Image, ImageDraw
import scipy.ndimage.morphology as morph

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from args import *


import json
with open('/storage/ProtopopovI/_data_/COCO/2014/annotations/person_keypoints_val2014.json') as data_file:    
    data_json = json.load(data_file)


class TrainPerson(Dataset):
    def __init__(self):
        super().__init__()
        self.target_trans = transforms.Compose([
            transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=0),
            transforms.ToTensor()
            ])
        self.search_trans = transforms.Compose([
            transforms.Resize((SEARCH_SIZE, SEARCH_SIZE), interpolation=0),
            transforms.ToTensor()
            ])
        self.file_names = sorted(os.listdir("/storage/ProtopopovI/_data_/COCO/2014/val2014/"))
    	
        
    def transform_score_label(self, depth2):
        depth2 = depth2.reshape(1, 1, depth2.size(0), depth2.size(1))
        max_value = depth2.max()
        depth2 = (depth2 == max_value).float()
        score_label = F.max_pool2d(depth2, kernel_size=(16, 16), padding=8, stride=16)
        score_zero = (score_label == 0).float()
        score_label = torch.stack([score_zero, score_label], dim=1).squeeze()
        return score_label

    def get_labels(self, object):
        labels = torch.tensor([])
        depths = torch.tensor([])
        score_labels = torch.tensor([])
        
        label1 = (object==0).float()
        depth1 = torch.tensor(morph.distance_transform_edt(np.asarray(label1[0])))
        label2 = (label1==0).float()
        depth2 = torch.tensor(morph.distance_transform_edt(np.asarray(label2[0])))
        depth = (depth1 + depth2).float().unsqueeze(0)
        label = torch.stack([label1, label2], dim=1)
        labels = torch.cat([labels, label], dim=0)
        depths = torch.cat([depths, depth], dim=0)
        score_label = self.transform_score_label(depth2).unsqueeze(0)
        score_labels = torch.cat([score_labels, score_label], dim=0)
        labels = labels.squeeze()
        score_labels = score_labels.squeeze()
        
        return labels, depths, score_labels