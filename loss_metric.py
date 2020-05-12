import torch

from args import *


########## Mask losses: ##########
def l2_loss(x, y, d):
    out = (d*(x - y)**2).sum()
    return out

def bce_loss(x, y, d):
    bce_loss =  torch.nn.BCELoss()
    return  bce_loss(x, y)

def dice_loss(x, y, d, smooth = 1.):
    
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    dice_loss = 1 - ((2*intersection + smooth) / (x_sum + y_sum + smooth))
    #print(dice_loss.mean().item())
    return dice_loss.mean()

def dice_combo_loss(x, y, d, bce_weight=0.5):
    dice_combo_loss = bce_weight * bce_loss(x, y, d) + (1 - bce_weight) * dice_loss(x, y, d)
    return dice_combo_loss

def l2_combo_loss(x, y, d):
    l2_combo_loss = l2_loss(x, y, d) * bce_loss(x, y, d)
    return l2_combo_loss


########## Score losses: ##########
def score_loss(xx, yy):
    bce_loss =  torch.nn.BCELoss()
    return  bce_loss(xx, yy)


########## All losses: ##########
def all_losses(x, y, d, xx, yy):
    x = x.reshape(-1, NUM_CLASSES, TARGET_SIZE, TARGET_SIZE)
    y = y.reshape(-1, NUM_CLASSES, TARGET_SIZE, TARGET_SIZE)
    d = d.reshape(-1, 1, TARGET_SIZE, TARGET_SIZE)
    xx, yy = xx.reshape(-1, 2, 17, 17), yy.reshape(-1, 2, 17, 17)
    all_losses =  dice_combo_loss(x, y, d) + 0.05*score_loss(xx, yy)
    return  all_losses