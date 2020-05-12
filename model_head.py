import torch
import torch.nn as nn
import torch.nn.functional as F

from args import *
from model_parts import *


'''
Model head
'''
class ModelDisigner(nn.Module):
	def __init__(self):
		super(ModelDisigner, self).__init__()
		self.backbone = BackboneUNet()
		self.score_branch = ScoreBranch()
		self.mask_branch = MaskBranch()
		# self.shape_branch = ShapeBranch()

		self.up_and_cat = UpAndCat()
		self.up = nn.Upsample(scale_factor=2, mode='nearest')
		self.up_conv_4 = nn.Sequential(
			ConvRelu(512+32, 512),
			ConvRelu(512, 512)
			)
		self.up_conv_3 = nn.Sequential(
			ConvRelu(512+256, 256),
			ConvRelu(256, 256)
			)
		self.up_conv_2 = nn.Sequential(
			ConvRelu(256+128, 128),
			ConvRelu(128, 128)
			)
		self.up_conv_1 = nn.Sequential(
			ConvRelu(128+64, 64),
			ConvRelu(64, 64)
			)
		self.final = nn.Sequential(
			nn.Conv2d(64, NUM_CLASSES, kernel_size=1),
			nn.Sigmoid()
			)

	def Correlation_func(self, s_f, t_f): # s_f-->search_feat, t_f-->target_feat
		s_f = s_f.reshape(TIMESTEPS, BATCH_SIZE, s_f.size(1), s_f.size(2), s_f.size(3)) # 5, 2, 256, 32, 32
		t_f = t_f.reshape(-1, 1, t_f.size(2), t_f.size(3)) # 512, 1, 32, 32
		outs = torch.tensor([]).to(device)
		for i in range(TIMESTEPS):
			out = s_f[i].reshape(1, -1, s_f[i].size(2), s_f[i].size(3)) # 1, 512, 32, 32
			out = F.conv2d(out, t_f, groups=out.size(1))
			out = out.reshape(BATCH_SIZE, s_f[i].size(1), out.size(2), out.size(3))
			outs = torch.cat([outs, out], dim=0)
		return outs

	def Chiose_RoW(self, corr_feat, pos_list):
		i_tensors = torch.tensor([]).to(device)
		for i in range(corr_feat.size(0)):
			j_tensors = torch.tensor([]).to(device)
			for j in range(corr_feat.size(1)):
				j_tensor = corr_feat[i][j][pos_list[i][j][0]][pos_list[i][j][1]].unsqueeze(0)
				j_tensors = torch.cat([j_tensors, j_tensor], dim=0)
			i_tensor = j_tensors.unsqueeze(0)
			i_tensors = torch.cat([i_tensors, i_tensor], dim=0)
		return i_tensors


	def Choise_feat(self, feat, pos_list, x):
		feat = feat.reshape(TIMESTEPS, BATCH_SIZE, feat.size(1), feat.size(2), feat.size(3))
		feat = feat.permute(0, 1, 3, 4, 2)

		i_tensors = torch.tensor([]).to(device)
		for i in range(feat.size(0)):
			j_tensors = torch.tensor([]).to(device)
			for j in range(feat.size(1)):
				j_tensor = feat[i][j][x*pos_list[i][j][0]:x*pos_list[i][j][0]+x*16, x*pos_list[i][j][1]:x*pos_list[i][j][1]+x*16, :].unsqueeze(0)
				j_tensors = torch.cat([j_tensors, j_tensor], dim=0)
			i_tensor = j_tensors.unsqueeze(0)
			i_tensors = torch.cat([i_tensors, i_tensor], dim=0)

		feat = i_tensors.permute(0, 1, 4, 2, 3)
		feat = feat.reshape(TIMESTEPS*BATCH_SIZE, feat.size(2), feat.size(3), feat.size(4))
		return feat


	def forward(self, target, searchs):
		_,  target_feat = self.backbone(target)
		searchs = searchs.permute(1, 0, 2, 3, 4).reshape(TIMESTEPS*BATCH_SIZE, INPUT_CHANNELS, SEARCH_SIZE, SEARCH_SIZE)
		search_cats, searchs_feat = self.backbone(searchs) 
		corr_feat = self.Correlation_func(searchs_feat, target_feat) # TIMESTEPS*BATCH_SIZE, 256, 17, 17

		##### Score Branch #####
		score, pos_list = self.score_branch(corr_feat) # score --> [TIMESTEPS*BATCH_SIZE, 2, 17, 17] # pos_list --> [TIMESTEPS*BATCH_SIZE, 2]
		score = score.reshape(TIMESTEPS, BATCH_SIZE, NUM_CLASSES, 17, 17)
		score = score.permute(1, 0, 2, 3, 4) # BATCH_SIZE, TIMESTEPS, NUM_CLASSES, 17, 17
		pos_list = pos_list.reshape(TIMESTEPS, BATCH_SIZE, 2) # TIMESTEPS, BATCH_SIZE, 2
		#out = self.shape_branch(corr_feat)

		##### Mask Branch #####
		corr_feat = corr_feat.reshape(TIMESTEPS, BATCH_SIZE, 256, 17, 17).permute(0, 1, 3, 4, 2) # TIMESTEPS, BATCH_SIZE, 256, 17, 17
		masks_feat = self.Chiose_RoW(corr_feat, pos_list) # TIMESTEPS, BATCH_SIZE, 256
		masks_feat = masks_feat.reshape(TIMESTEPS*BATCH_SIZE, 256, 1, 1) # TIMESTEPS*BATCH_SIZE, 256, 1, 1
		masks_feat = self.mask_branch(masks_feat) # TIMESTEPS*BATCH_SIZE, 32, 16, 16
		'''
		search_cats.Size -- > TIMESTEPS*BATCH_SIZE, CHANNELS, SIZE, SIZE

		search_cats[3].shape:  torch.Size([10, 512, 32, 32])
		search_cats[2].shape:  torch.Size([10, 256, 64, 64])
		search_cats[1].shape:  torch.Size([10, 128, 128, 128])
		search_cats[0].shape:  torch.Size([10, 64, 256, 256])

		masks_feat.Size -- > TIMESTEPS*BATCH_SIZE, CHANNELS, SIZE, SIZE

		masks_feat.shape:      torch.Size([10, 32, 16, 16])
		'''
		feat = self.Choise_feat(search_cats[3], pos_list, 1)
		masks = torch.cat([masks_feat, feat], dim=1)
		masks = self.up_conv_4(masks)
		masks = self.up(masks)

		feat = self.Choise_feat(search_cats[2], pos_list, 2)
		masks = torch.cat([masks, feat], dim=1)
		masks = self.up_conv_3(masks)
		masks = self.up(masks)
		
		feat = self.Choise_feat(search_cats[1], pos_list, 4)
		masks = torch.cat([masks, feat], dim=1)
		masks = self.up_conv_2(masks)
		masks = self.up(masks)

		feat = self.Choise_feat(search_cats[0], pos_list, 8)
		masks = torch.cat([masks, feat], dim=1)
		masks = self.up_conv_1(masks)

		masks = self.final(masks) # masks.shape:  torch.Size([10, 2, 128, 128])
		masks = masks.reshape(TIMESTEPS, BATCH_SIZE, NUM_CLASSES, 128, 128).permute(1, 0, 2, 3, 4) # BATCH_SIZE, TIMESTEPS, 2, 128, 128
		return score, masks

if __name__ == '__main__':
	model = ModelDisigner()
	model = model.to(device)
	target = torch.rand([BATCH_SIZE, 3, 128, 128]).to(device)
	searchs = torch.rand([BATCH_SIZE, TIMESTEPS, 3, 256, 256]).to(device)
	score, masks = model(target, searchs)
	print('score.shape: ', score.shape)
	print('masks.shape: ', masks.shape)
