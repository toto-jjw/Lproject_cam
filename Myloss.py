# Myloss.py (Final Integrated Version with DPCE's Loss Implementations)

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import torchvision.models as models
import torchvision.transforms as transforms
import random
import numpy as np

# --- 1. 기본 구성 요소 손실 ---

class GradientConsistencyLoss(nn.Module):
    """ L_stereo와 L_depth의 기본 연산 (변경 없음) """
    def __init__(self):
        super(GradientConsistencyLoss, self).__init__()
        kernel_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernels = [kernel_left, kernel_right, kernel_up, kernel_down]
        self.pool = nn.AvgPool2d(4)

    def forward(self, img1, img2):
        img1_pool, img2_pool = self.pool(img1), self.pool(img2)
        total_loss = 0
        for kernel in self.kernels:
            kernel = kernel.to(img1.device)
            grad1 = F.conv2d(img1_pool, kernel.repeat(img1.shape[1], 1, 1, 1), padding=1, groups=img1.shape[1])
            grad2 = F.conv2d(img2_pool, kernel.repeat(img2.shape[1], 1, 1, 1), padding=1, groups=img2.shape[1])
            total_loss += F.l1_loss(torch.abs(grad1), torch.abs(grad2))
        return total_loss / len(self.kernels)

# --- 2. ★★★ DPCE-Net 방식의 손실 함수들로 교체 ★★★ ---

class LightConsistencyLoss(nn.Module):
    """ L_light: DPCE-Net 방식 (Global + Local L2 Loss) """
    def __init__(self, patch_size=32, num_patches=10, target_L=0.6, lambda_L=4.5):
        super(LightConsistencyLoss, self).__init__()
        self.target_L = target_L
        self.lambda_L = lambda_L
        self.patch_size = patch_size
        self.num_patches = num_patches

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
                max_y, max_x = H - self.patch_size, W - self.patch_size
                for _ in range(self.num_patches):
                    rand_y = random.randint(0, max_y)
                    rand_x = random.randint(0, max_x)
                    patch = img[:, rand_y : rand_y + self.patch_size, rand_x : rand_x + self.patch_size]
                    all_sampled_patches.append(patch)
            sampled_patches_tensor = torch.stack(all_sampled_patches, dim=0)
            local_mean_patches = torch.mean(sampled_patches_tensor, dim=[2, 3])
            loss_local = torch.mean(torch.pow(local_mean_patches - self.target_L, 2))

        return loss_global + self.lambda_L * loss_local

class SpatialConsistencyLoss(nn.Module):
    """ L_sfp: DPCE-Net 방식 (VGG11, relu3_2, MSE, Normalize) """
    def __init__(self, device):
        super(SpatialConsistencyLoss, self).__init__()
        vgg_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg_model.children())[:15]).to(device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.loss_fn = nn.MSELoss()

    def forward(self, original_image, enhanced_image):
        norm_input = self.normalize(original_image)
        norm_enhanced = self.normalize(enhanced_image)
        features_input = self.feature_extractor(norm_input)
        features_enhanced = self.feature_extractor(norm_enhanced)
        return self.loss_fn(features_input, features_enhanced)

class GammaSmoothnessLoss(nn.Module):
    """ L_gamma: DPCE-Net 방식 (2-pixel-gap L2 Loss) """
    def __init__(self):
        super(GammaSmoothnessLoss, self).__init__()

    def forward(self, gamma_map):
        grad_x_sq = torch.pow(gamma_map[:, :, :, 2:] - gamma_map[:, :, :, :-2], 2)
        grad_y_sq = torch.pow(gamma_map[:, :, 2:, :] - gamma_map[:, :, :-2, :], 2)
        return (torch.mean(grad_x_sq) + torch.mean(grad_y_sq)) / 2


# --- ★★★ Color Ratio Preservation Loss (원본 색상 비율 유지) ★★★ ---

class ColorConsistencyLoss(nn.Module):
    """
    픽셀별 색상 비율 보존 손실
    
    ★★★ 핵심 아이디어 ★★★
    - 각 픽셀에서 색상 비율(R:G:B)을 계산
    - 원본과 Enhanced의 픽셀별 색상 비율이 일치하도록 강제
    - 결과: 각 픽셀의 색상 톤 보존
    """
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()
        self.eps = 1e-6
    
    def forward(self, original, enhanced):
        """
        Args:
            original: 원본 이미지 [B, 3, H, W]
            enhanced: Enhanced 이미지 [B, 3, H, W]
        Returns:
            픽셀별 색상 비율 차이 손실
        """
        if original.shape[1] == 1:
            # Grayscale 모드면 손실 없음
            return torch.tensor(0.0, device=original.device)
        
        # 원본의 픽셀별 색상 비율: R/(R+G+B), G/(R+G+B), B/(R+G+B)
        orig_sum = original.sum(dim=1, keepdim=True) + self.eps
        orig_ratio = original / orig_sum  # [B, 3, H, W]
        
        # Enhanced의 픽셀별 색상 비율
        enh_sum = enhanced.sum(dim=1, keepdim=True) + self.eps
        enh_ratio = enhanced / enh_sum  # [B, 3, H, W]
        
        # 픽셀별 비율 차이 최소화
        return F.mse_loss(orig_ratio, enh_ratio)
# --- 3. ★★★ 최종 통합 손실 함수 (DPCE 방식 + Color Loss) ★★★
class DimCamLoss(nn.Module):
    def __init__(self, device, lambda_stereo=1.0, lambda_depth=1.0, 
                 w_light=1.0, w_sfp=2.0, w_gamma=0.01, w_color=0.5, **kwargs):
        super(DimCamLoss, self).__init__()
        
        self.lambda_stereo = lambda_stereo
        self.lambda_depth = lambda_depth
        self.w_light = w_light
        self.w_sfp = w_sfp
        self.w_gamma = w_gamma
        self.w_color = w_color  # ★ Color Loss 가중치
        
        # DPCE-Net 방식의 손실 함수들로 인스턴스화
        self.loss_grad_consistency = GradientConsistencyLoss()
        self.loss_light = LightConsistencyLoss(**kwargs)
        self.loss_sfp = SpatialConsistencyLoss(device)
        self.loss_gamma = GammaSmoothnessLoss()
        self.loss_color = ColorConsistencyLoss()  # ★ Color Loss 추가

    def _get_homography_from_calib(self, calib_data, device):
        # (변경 없음)
        try:
            K_l_mat = calib_data['left_intrinsics']['camera_matrix']
            K_r_mat = calib_data['right_intrinsics']['camera_matrix']
            K_l = torch.from_numpy(K_l_mat).float().to(device)
            K_r = torch.from_numpy(K_r_mat).float().to(device)
            R_mat = calib_data['extrinsics']['rotation_matrix']
            R_lr = torch.from_numpy(R_mat).float().to(device)
            K_r_inv = torch.inverse(K_r)
            H_l_from_r = K_l @ R_lr @ K_r_inv
            return H_l_from_r
        except (KeyError, TypeError, AttributeError, np.linalg.LinAlgError):
            return torch.eye(3, device=device)

    def forward(self, original_l, original_r, pred_l, pred_r, 
                fused_gamma_l, fused_gamma_r, depth_map, calib_data):
        device = pred_l.device
        
        # --- 1. Stereo Consistency Loss (L_stereo) ---
        loss_s = torch.tensor(0.0, device=device)
        if self.lambda_stereo > 0:
            H_l_from_r = self._get_homography_from_calib(calib_data, device)
            H_l_from_r_b = H_l_from_r.unsqueeze(0).expand(pred_l.size(0), -1, -1)
            
            mask_r = torch.ones_like(pred_r)
            mask_warped = kornia.geometry.transform.warp_perspective(mask_r, H_l_from_r_b, 
                                                                    (pred_l.shape[2], pred_l.shape[3]))
            mask_warped = (mask_warped > 0.99).float()

            pred_r_warped = kornia.geometry.transform.warp_perspective(pred_r, H_l_from_r_b, 
                                                                        (pred_l.shape[2], pred_l.shape[3]))
            loss_s = self.loss_grad_consistency(pred_l * mask_warped, pred_r_warped * mask_warped)

        # --- 2. Depth Consistency Loss (L_depth) ---
        loss_d = torch.tensor(0.0, device=device)
        if self.lambda_depth > 0 and depth_map is not None:
            pred_l_gray = kornia.color.rgb_to_grayscale(pred_l)
            with torch.no_grad():
                B, C, H, W = depth_map.shape
                d_min = depth_map.view(B, -1).min(dim=1, keepdim=True)[0]
                d_max = depth_map.view(B, -1).max(dim=1, keepdim=True)[0]
                depth_norm = (depth_map.view(B, -1) - d_min) / (d_max - d_min + 1e-8)
                depth_norm = depth_norm.view(B, C, H, W)
            loss_d = self.loss_grad_consistency(pred_l_gray, depth_norm)

        # --- 3. Light Consistency Loss (L_light) ---
        loss_light = self.loss_light(pred_l) + self.loss_light(pred_r)

        # --- 4. Spatial Consistency Loss (L_sfp) ---
        loss_sfp = self.loss_sfp(original_l, pred_l) + self.loss_sfp(original_r, pred_r)

        # --- 5. Gamma Smoothness Loss (L_gamma) ---
        loss_gamma = self.loss_gamma(fused_gamma_l) + self.loss_gamma(fused_gamma_r)

        # --- 6. ★★★ Color Ratio Preservation Loss ★★★ ---
        loss_color = torch.tensor(0.0, device=device)
        if self.w_color > 0:
            loss_color = (self.loss_color(original_l, pred_l) + 
                         self.loss_color(original_r, pred_r))

        # --- 최종 손실 계산 ---
        total_loss = (self.lambda_stereo * loss_s + 
                      self.lambda_depth * loss_d +
                      self.w_light * loss_light +
                      self.w_sfp * loss_sfp +
                      self.w_gamma * loss_gamma +
                      self.w_color * loss_color)
        
        loss_dict = {
            "total": total_loss.item(),
            "stereo": loss_s.item(),
            "depth": loss_d.item(),
            "light": loss_light.item(),
            "sfp": loss_sfp.item(),
            "gamma": loss_gamma.item(),
            "color": loss_color.item(),
        }
        
        return total_loss, loss_dict

