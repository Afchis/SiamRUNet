import torch


BATCH_SIZE = 16
UNET_CLASSES = 1

DEVICE = "cuda:0"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
device
