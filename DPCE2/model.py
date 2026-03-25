import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np


def gsigmoid(x, k):
    # x, k: torch.Tensor 혹은 float, 단 k>0 가정
    sigmoid_2 = 2 / (1 + torch.exp(-x))
    out = torch.where(
        x <= 0,
        sigmoid_2 * (1 - k) + k,
        sigmoid_2 * (1/k - 1) + 2 - 1/k
    )
    return out


def gamma_enhance(x, gamma):
    """
    x: torch.Tensor, normalized [0,1] (이미지 가능, shape 상관없음)
    gamma: float or torch.float, 강화계수
    """
    x_gamma = torch.pow(x, gamma)
    cond = x_gamma > (1 - x)
    val_if_true = x_gamma
    val_if_false = -torch.pow(1 - x, 1.0 / gamma) + 1
    out = torch.where(cond, val_if_true, val_if_false)
    return out


class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,8,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(8,32,3,2,1,bias=True) 
		self.e_conv3 = nn.Conv2d(32,32,3,2,1,bias=True) 
		self.e_conv4 = nn.Conv2d(64,8,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(16,3,3,1,1,bias=True) 

		self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
		self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.upsample1(x3)

		x5 = self.relu(self.e_conv4(torch.cat([x2,x4],1)))
		x6 = self.upsample1(x5)
		gamma = gsigmoid(self.e_conv5(torch.cat([x1,x6],1)), k=0.1)

		enhance_image = gamma_enhance(x, gamma)

		return enhance_image,gamma



