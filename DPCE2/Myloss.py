import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import vgg11, VGG11_Weights
import numpy as np
import random
import torchvision.transforms as transforms

class L_light(nn.Module):
    def __init__(self, patch_size=32, num_patches=10, target_L=0.6, lambda_L=4.5):
        super(L_light, self).__init__()
        self.target_L = target_L
        self.lambda_L = lambda_L
        self.patch_size = patch_size
        self.num_patches = num_patches
        print(f"L_light initialized with (safe random sampling): "
              f"patch_size={patch_size}, num_patches={num_patches}, "
              f"target_L={target_L}, lambda_L={lambda_L}")

    def forward(self, enhanced_image):
        global_mean_rgb = torch.mean(enhanced_image, dim=[2, 3])
        loss_global = torch.mean(torch.pow(global_mean_rgb - self.target_L, 2))

        B, C, H, W = enhanced_image.shape
        
        if H < self.patch_size or W < self.patch_size:
            loss_local = torch.tensor(0.0, device=enhanced_image.device)
        else:
            all_sampled_patches = []
            for i in range(B):
                img = enhanced_image[i]
                max_y = H - self.patch_size
                max_x = W - self.patch_size
                for _ in range(self.num_patches):
                    rand_y = random.randint(0, max_y)
                    rand_x = random.randint(0, max_x)
                    patch = img[:, rand_y : rand_y + self.patch_size, rand_x : rand_x + self.patch_size]
                    all_sampled_patches.append(patch)
            
            sampled_patches_tensor = torch.stack(all_sampled_patches, dim=0)
            local_mean_patches = torch.mean(sampled_patches_tensor, dim=[2, 3])
            loss_local = torch.mean(torch.pow(local_mean_patches - self.target_L, 2))

        total_loss = loss_global + self.lambda_L * loss_local
        return total_loss

class L_gamma(nn.Module):
    def __init__(self):
        super(L_gamma, self).__init__()

    def forward(self, gamma_map):
        grad_x_sq = torch.pow(gamma_map[:, :, :, 2:] - gamma_map[:, :, :, :-2], 2)
        grad_y_sq = torch.pow(gamma_map[:, :, 2:, :] - gamma_map[:, :, :-2, :], 2)
        loss = (torch.mean(grad_x_sq) + torch.mean(grad_y_sq)) / 2
        return loss
    
class L_sfp(nn.Module):
    def __init__(self):
        super(L_sfp, self).__init__()
        vgg_model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg_model.children())[:15]) # relu4_2 특징 추출 relu3_2는 [:10]
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        self.loss_fn = nn.MSELoss()

    def forward(self, input_image, enhanced_image):
        norm_input = self.normalize(input_image)
        norm_enhanced = self.normalize(enhanced_image)

        features_input = self.feature_extractor(norm_input)
        features_enhanced = self.feature_extractor(norm_enhanced)
        
        loss = self.loss_fn(features_input, features_enhanced)
        return loss

class TotalLoss(nn.Module):
    def __init__(self, 
                 light_patch_size=32, light_num_patches=10, light_target_L=0.6, light_lambda_L=4.5,
                 w_light=1.0, w_sfp=2.0, w_gamma=0.01):
        super(TotalLoss, self).__init__()
        
        self.loss_light = L_light(light_patch_size, light_num_patches, light_target_L, light_lambda_L)
        self.loss_sfp = L_sfp()
        self.loss_gamma = L_gamma()

        self.w_light = w_light
        self.w_sfp = w_sfp
        self.w_gamma = w_gamma

    def forward(self, original_image, enhanced_image, gamma_map):
        loss_light = self.loss_light(enhanced_image)
        loss_sfp = self.loss_sfp(original_image, enhanced_image)
        loss_gamma = self.loss_gamma(gamma_map)

        total_loss = (self.w_light * loss_light +
                      self.w_sfp * loss_sfp + 
                      self.w_gamma * loss_gamma)

        loss_components = {
            'total_loss': total_loss.item(),
            'loss_light': loss_light.item(),
            'loss_sfp': loss_sfp.item(),
            'loss_gamma': loss_gamma.item(),
        }
        return total_loss, loss_components
