import json
import os
import numpy as np

from PIL import Image, ImageDraw
import scipy.ndimage.morphology as morph

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from args import *


data_path = "/storage/ProtopopovI/_data_/ytb_vos/train/JPEGImages/"


with open('/storage/ProtopopovI/_project_/SiamRUNet/test.json') as data_file:
    data_json = json.load(data_file)


class YtbVosData(Dataset):
    def __init__(self):
        super().__init__()
        self.json = data_json
        self.data_path = "/storage/ProtopopovI/_data_/ytb_vos/train/JPEGImages/"

        self.trans = transforms.Compose([
            transforms.Resize((256, 256), interpolation=0),
            transforms.ToTensor()
            ])
        
    def Rle_to_numpy(self, RLE, width, height):
        NOT_RLE = []
        try:
            for i, data in enumerate(RLE):
                if i % 2 == 0:
                    x = 0
                else:
                    x = 1
                for j in range(data):
                    NOT_RLE.append(x)
            np_array = np.asarray(NOT_RLE)
            np_array = np_array.reshape(width, height).T#.tolist()
            np_array = np.uint8(np_array*255)
        except TypeError:
            np_array = np.zeros((width, height))
        return np_array
    
    def relative_bbox(self, center, bbox, width, height):
        try:
            re_center = [center[0]/width, center[1]/height]
            re_bbox = [
                bbox[0]/width,
                bbox[1]/height,
                bbox[2]/width,
                bbox[3]/height,
            ]
        except TypeError:
            re_center = [0, 0]
            re_bbox = [
                0,
                0,
                0,
                0,
            ]
        return re_center, re_bbox
        
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

    def Choise_feat(self, label, score_label, x=8):
        score_label = score_label[0][1]
        max_value = score_label.max()
        pos = (score_label == max_value).nonzero()#.unsqueeze(0)

        label = label.permute(0, 2, 3, 1)
        i_tensors = torch.tensor([])
        for i in range(label.size(0)):
            i_tensor = label[i][x*pos[i][0]:x*pos[i][0]+x*16, x*pos[i][1]:x*pos[i][1]+x*16, :].unsqueeze(0)
            i_tensors = torch.cat([i_tensors, i_tensor], dim=0)

        label = i_tensors.permute(0, 3, 1, 2)
        return label
    
    def __len__(self):
        return len(self.json)
    
    def  __getitem__(self, idx):
        search_name = self.json[idx]['file_name']
        target_name = self.json[idx]['target_info']['target_name']
        RLE_search = data_json[idx]['segmentation']
        width = data_json[idx]['width']
        height = data_json[idx]['height']
        search = Image.open(self.data_path + search_name).convert('RGB')
        seg_search = self.Rle_to_numpy(RLE_search, width, height)
        mask_search = Image.fromarray(seg_search)
        target = Image.open(self.data_path + target_name).convert('RGB')
        
        sr_center, sr_bbox = self.relative_bbox(self.json[idx]['center'], self.json[idx]['bbox'], width, height)
        tr_center, _ = self.relative_bbox(self.json[idx]['target_info']['target_center'], 
                                          self.json[idx]['target_info']['target_bbox'], width, height)
        
        search = self.trans(search)
        mask_search = self.trans(mask_search)
        target = self.trans(target)
        axis_x = round(tr_center[0]*256)
        if axis_x < 64:
            axis_x = 64
        elif axis_x > 192:
            axis_x = 192
        axis_y = round(tr_center[1]*256)
        if axis_y < 64:
            axis_y = 64
        elif axis_y > 192:
            axis_y = 192
        target = target[:, axis_y-64:axis_y+64, axis_x-64:axis_x+64]
        
        # axis_x = round(sr_center[0]*256)
        # if axis_x < 64:
        #     axis_x = 64
        # elif axis_x > 192:
        #     axis_x = 192
        # axis_y = round(sr_center[1]*256)
        # if axis_y < 64:
        #     axis_y = 64
        # elif axis_y > 192:
        #     axis_y = 192
        # mask_search = mask_search[:, axis_y-64:axis_y+64, axis_x-64:axis_x+64]
        
        label, depth, score_label = self.get_labels(mask_search)
        search, label, depth, score_label = search.unsqueeze(0), label.unsqueeze(0), depth.unsqueeze(0), score_label.unsqueeze(0)
        label = self.Choise_feat(label, score_label)
        depth = self.Choise_feat(depth, score_label)
        
        #return search, mask_search, search_name, target, target_name
        return target, search, label, depth, score_label


train_dataset = YtbVosData()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=1,
                          shuffle=True)


if __name__ == '__main__':
	# print('Write number of image in dataset: ')
	# inp = int(input())
	target, search, label, depth, score_label = train_dataset[9]
	print('target.shape', target.shape)
	print('search.shape', search.shape)
	print('label.shape', label.shape)
	print('depth.shape', depth.shape)
	print('score_label.shape', score_label.shape)
	# print(score_label)