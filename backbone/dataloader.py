import sys
sys.path.insert(0, "..")
from args import *

import os
import numpy as np

from PIL import Image
import scipy.ndimage.morphology as morph

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


class TrainData(Dataset):
	def __init__(self):
		super().__init__()
		self.img_path = "/storage/ProtopopovI/_data_/COCO/COCO/images/train2017/"
		self.msk_path = "/storage/ProtopopovI/_data_/COCO/COCO/annotations/train2017/"
		self.transform = transforms.Compose([
			transforms.Resize((256, 256), interpolation=0),
			transforms.ToTensor()
			])

	def img_names(self):
		return sorted(os.listdir(self.img_path))

	def msk_names(self):
		return sorted(os.listdir(self.msk_path))

	def get_labels(self, masks, num_classes=UNET_CLASSES):
		labels = torch.tensor([])
		depths = torch.zeros([256, 256])
		for i in range(num_classes):
			label = (masks==i).float()
			depth = torch.tensor(morph.distance_transform_edt(np.asarray(label[0])))
			labels = torch.cat([labels, label], dim=0)
			depths += depth
		return labels, depths

	def __getitem__(self, idx):
		img_names = self.img_names()
		msk_names = self.msk_names()

		image = self.transform(Image.open(self.img_path + img_names[idx]))
		masks = self.transform(Image.open(self.msk_path + msk_names[idx]))*255
		labels, depths = self.get_labels(masks)
		return image, labels, depths

	def __len__(self):
		return len(self.img_names())

train_dataset = TrainData()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=1,
                          shuffle=True)



if __name__ == '__main__':
	data = TrainData()
	image, labels, depths = data[0]
	print(image.shape, labels.shape, depths.shape)
	print(depths[20])