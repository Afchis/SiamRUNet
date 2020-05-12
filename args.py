import torch


BATCH_SIZE = 8
TIMESTEPS = 1
INPUT_CHANNELS = 3
NUM_CLASSES = 2
SEARCH_SIZE = 256
TARGET_SIZE = 128
UNET_NUM_CLASSES = 2

MODE = 'Standart'
CELL_MODEL = 'Rnn'

UNET_WEIGHTS = "pathignore/weights/weights.pth"

DEVICE = "cuda:1"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
device

# tensorboard --logdir=runs
