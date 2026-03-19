import sys, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
IMG_SIZE = 32 #according to CIFAR 10 
PATCH = 4 # means that the patch size is4x4 and thus total patches will be (32/4)^2 = 64
N_CLASSES=10
EMBED = 256 #TOKEN EMBEDDING DIMENSIONS
DEPTH = 6 #NO OF TRANSFORMER BLOCKS
HEADS = 4 #NO. OF ATTENTION HEADS
MLP_RATIO = 4
DROP = 0.1
BATCH = 512
LR = 3e-5
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "nanaovit.pt"    
#-----creating patch embeddings-------#
class PatchEmbedd(nn.Module):
    def __init__(self):
        super().__init__()
        n_patches=(IMG_SIZE//PATCH)**2  #64
        patch_dim=3*PATCH**2 # 3*4*4=48
        self.proj = nn.Conv2d(3,EMBED,PATCH,stride=PATCH) #use a conv2d layer to cut the image and flatten them in a single step
        self.n=n_patches

    def forward(self,x):
        x=self.proj(x)
        x=x.flatten(2).transpose(1,2)
        return x
#-------------MHA---------------------------------------------#


