import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from class_set_gtnnwr import Model
import torch

print(torch.cuda.is_available())