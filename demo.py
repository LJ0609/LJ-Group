from trans import ViT
import cv2 as cv
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

v = ViT(
    #patch_size = 16,
    num_classes = 8,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
#img = torch.randn(1, 3, 224, 224)
img = torch.randn(1, 10, 12, 1024)
print(img.shape)
# preds = v(img) # (1, 1000)
fc=nn.Linear(1024,256)
preds=fc(img)
print(preds)
print(preds.shape) #[1,145,10]