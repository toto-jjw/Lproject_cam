# Lproject_cam/Myloss.py
# Loss Functions for DimCam with FoundationStereo
# FoundationStereo의 disparity 기반 depth와 호환되도록 수정

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
    """L_stereo와 L_depth의 기본 연산"""
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


# --- 2. Adaptive Light Consistency Loss ---

class LightConsistencyLoss(nn.Module):
    """
    L_light: DPCE-Net 방식 (Global + Local L2 Loss)
    ★★★ 고정 목표값 target_L=0.6 사용 (기존 DimCam2와 동일) ★★★
    """
    def __init__(self, patch_size=32, num_patches=10, target_L=0.6, lambda_L=4.5, **kwargs):
        super(LightConsistencyLoss, self).__init__()
        self.target_L = target_L
        self.lambda_L = lambda_L
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self, enhanced_image):
        """enhanced_image만 입력받음 (original 불필요)"""
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
                    patch = img[:, rand_y:rand_y + self.patch_size, rand_x:rand_x + self.patch_size]
                    all_sampled_patches.append(patch)
            
            sampled_patches_tensor = torch.stack(all_sampled_patches, dim=0)
            local_mean_patches = torch.mean(sampled_patches_tensor, dim=[2, 3])
            loss_local = torch.mean(torch.pow(local_mean_patches - self.target_L, 2))

        return loss_global + self.lambda_L * loss_local


# --- 3. Spatial Consistency Loss (Reference-based) ---

class SpatialConsistencyLoss(nn.Module):
    """
    L_sfp: DPCE-Net 방식 (VGG11, relu3_2, MSE, Normalize)
    ★★★ Reference-based: 극저조도에서는 200ms 참조 이미지 사용 ★★★
    """
    def __init__(self, device):
        super(SpatialConsistencyLoss, self).__init__()
        vgg_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg_model.children())[:15]).to(device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.loss_fn = nn.MSELoss()

    def forward(self, reference_image, enhanced_image):
        """
        Args:
            reference_image: 참조 이미지 (극저조도: 200ms, 중간노출: 원본)
            enhanced_image: 향상된 이미지
        """
        norm_ref = self.normalize(reference_image)
        norm_enhanced = self.normalize(enhanced_image)
        features_ref = self.feature_extractor(norm_ref)
        features_enhanced = self.feature_extractor(norm_enhanced)
        return self.loss_fn(features_ref, features_enhanced)


# --- EnhancedPerceptualLoss는 더 이상 사용하지 않음 (호환성 유지) ---

class EnhancedPerceptualLoss(nn.Module):
    """다중 스케일 VGG 특징 사용"""
    def __init__(self, device):
        super(EnhancedPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])
        self.slice4 = nn.Sequential(*list(vgg.children())[16:23])
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.weights = [1.0, 0.75, 0.5, 0.25]

    def forward(self, original_image, enhanced_image):
        norm_orig = self.normalize(original_image)
        norm_enh = self.normalize(enhanced_image)
        
        total_loss = 0
        
        f1_orig = self.slice1(norm_orig)
        f1_enh = self.slice1(norm_enh)
        total_loss += self.weights[0] * F.mse_loss(f1_orig, f1_enh)
        
        f2_orig = self.slice2(f1_orig)
        f2_enh = self.slice2(f1_enh)
        total_loss += self.weights[1] * F.mse_loss(f2_orig, f2_enh)
        
        f3_orig = self.slice3(f2_orig)
        f3_enh = self.slice3(f2_enh)
        total_loss += self.weights[2] * F.mse_loss(f3_orig, f3_enh)
        
        f4_orig = self.slice4(f3_orig)
        f4_enh = self.slice4(f3_enh)
        total_loss += self.weights[3] * F.mse_loss(f4_orig, f4_enh)
        
        return total_loss


# --- 4. Gamma Smoothness Loss ---

class GammaSmoothnessLoss(nn.Module):
    """감마 맵 공간적 평활화"""
    def __init__(self):
        super(GammaSmoothnessLoss, self).__init__()

    def forward(self, gamma_map):
        grad_x_sq = torch.pow(gamma_map[:, :, :, 2:] - gamma_map[:, :, :, :-2], 2)
        grad_y_sq = torch.pow(gamma_map[:, :, 2:, :] - gamma_map[:, :, :-2, :], 2)
        return (torch.mean(grad_x_sq) + torch.mean(grad_y_sq)) / 2


# --- 5. ★★★ Color Ratio Preservation Loss (원본 색상 비율 유지) ★★★ ---

class ColorConsistencyLoss(nn.Module):
    """
    픽셀별 색상 비율 보존 손실
    
    ★★★ 핵심 아이디어 ★★★
    - 각 픽셀에서 색상 비율(R:G:B)을 계산
    - 원본과 Enhanced의 픽셀별 색상 비율이 일치하도록 강제
    - 결과: 각 픽셀의 색상 톤 보존
    
    Input: original image, enhanced image
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


# --- 7. ★★★ FoundationStereo 호환 Depth Consistency Loss ★★★ ---

class DepthConsistencyLoss(nn.Module):
    """
    Depth Consistency Loss - 200ms 기준 Depth를 GT로 활용
    
    ★★★ 핵심 아이디어 ★★★
    - 200ms 이미지로 계산된 depth = "GT 구조 정보" (물체 경계 위치)
    - 목표: 모든 노출도(001ms~500ms) 이미지를 200ms 품질로 향상
    - 따라서 모든 enhanced 이미지는 200ms depth의 edge 구조를 따라야 함
    - ★★★ 왼쪽/오른쪽 이미지 모두에 depth 손실 적용 ★★★
    
    손실 구성:
    1. Edge Direction Consistency: enhanced edge 방향이 depth edge 방향과 일치
    2. Edge Presence: depth edge 위치에 enhanced에도 edge가 존재
    """
    def __init__(self, edge_threshold=0.1):
        super(DepthConsistencyLoss, self).__init__()
        self.edge_threshold = edge_threshold
        
        # Sobel 필터 정의
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)

    def _compute_gradient(self, img):
        """Sobel gradient 계산"""
        if img.shape[1] == 3:
            img = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        
        return grad_x, grad_y

    def _compute_single_image_loss(self, enhanced_img, depth_map):
        """단일 이미지에 대한 depth consistency loss 계산"""
        # 1. Gradient 계산
        depth_grad_x, depth_grad_y = self._compute_gradient(depth_map)
        enh_grad_x, enh_grad_y = self._compute_gradient(enhanced_img)
        
        # 2. Gradient magnitude
        depth_mag = torch.sqrt(depth_grad_x**2 + depth_grad_y**2 + 1e-8)
        enh_mag = torch.sqrt(enh_grad_x**2 + enh_grad_y**2 + 1e-8)
        
        # 3. Depth edge mask (200ms GT의 edge 위치)
        depth_mag_norm = depth_mag / (depth_mag.max() + 1e-8)
        edge_mask = (depth_mag_norm > self.edge_threshold).float()
        
        # 4. Edge Direction Consistency Loss
        depth_dir_x = depth_grad_x / depth_mag
        depth_dir_y = depth_grad_y / depth_mag
        enh_dir_x = enh_grad_x / enh_mag
        enh_dir_y = enh_grad_y / enh_mag
        
        direction_similarity = torch.abs(depth_dir_x * enh_dir_x + depth_dir_y * enh_dir_y)
        direction_loss = (1.0 - direction_similarity) * edge_mask
        direction_loss = direction_loss.sum() / (edge_mask.sum() + 1e-8)
        
        # 5. Edge Presence Loss
        enh_mag_norm = enh_mag / (enh_mag.max() + 1e-8)
        presence_loss = F.relu(self.edge_threshold - enh_mag_norm) * edge_mask
        presence_loss = presence_loss.sum() / (edge_mask.sum() + 1e-8)
        
        return 0.7 * direction_loss + 0.3 * presence_loss

    def forward(self, enhanced_l, enhanced_r, depth_map_l, depth_map_r, exposure=None):
        """
        ★★★ 왼쪽과 오른쪽 이미지 각각에 해당 depth map 적용 ★★★
        
        Args:
            enhanced_l: Enhanced left image [B, 3, H, W]
            enhanced_r: Enhanced right image [B, 3, H, W]  
            depth_map_l: Left depth map from L→R stereo [B, 1, H, W]
            depth_map_r: Right depth map from R→L stereo [B, 1, H, W]
            exposure: (unused, kept for API compatibility)
        
        Returns:
            Combined edge-aware depth consistency loss for both L/R images
        
        ★ 각 이미지에 정확한 depth map 적용 (시차 문제 해결)
        """
        loss_l = torch.tensor(0.0, device=enhanced_l.device)
        loss_r = torch.tensor(0.0, device=enhanced_r.device)
        
        # 왼쪽 이미지 + 왼쪽 depth
        if depth_map_l is not None:
            loss_l = self._compute_single_image_loss(enhanced_l, depth_map_l)
        
        # 오른쪽 이미지 + 오른쪽 depth
        if depth_map_r is not None:
            loss_r = self._compute_single_image_loss(enhanced_r, depth_map_r)
        
        # 유효한 손실만 평균
        if depth_map_l is not None and depth_map_r is not None:
            total_loss = (loss_l + loss_r) / 2.0
        elif depth_map_l is not None:
            total_loss = loss_l
        elif depth_map_r is not None:
            total_loss = loss_r
        else:
            total_loss = torch.tensor(0.0, device=enhanced_l.device)
        
        return total_loss


# --- 6. ★★★ 통합 손실 함수 ★★★ ---

class DimCamLoss(nn.Module):
    """
    DimCam Loss - Reference-based Version
    
    ★★★ 핵심 개선: Exposure-aware Loss Weighting ★★★
    1. 극저조도 (001ms~010ms): SFP는 200ms 참조 이미지와 비교
    2. 중간 노출 (050ms~200ms): SFP는 원본과 비교
    3. Light Loss: 노출도에 따라 가중치 조절
    
    ★★★ 손실 구성 ★★★
    1. Stereo Consistency: 좌/우 향상 결과 일관성
    2. Depth Consistency: 깊이 edge와 향상 결과 일관성  
    3. Light: 목표 밝기 도달 (target_L=0.6)
    4. Perceptual (SFP): 구조 보존 (Reference-based)
    5. Gamma Smoothness: 감마 맵 평활화
    6. Color: RGB 채널 간 색상 비율 보존
    """
    def __init__(self, device, 
                 lambda_stereo=2.0, lambda_depth=0.1,
                 w_light=0.1,
                 w_sfp=0.2,
                 w_gamma=0.001,
                 w_color=0.5,
                 use_reference_sfp=True,  # ★ Reference-based SFP 활성화
                 **kwargs):
        super(DimCamLoss, self).__init__()
        
        self.lambda_stereo = lambda_stereo
        self.lambda_depth = lambda_depth
        self.w_light = w_light
        self.w_sfp = w_sfp
        self.w_gamma = w_gamma
        self.w_color = w_color
        self.use_reference_sfp = use_reference_sfp
        
        # ★★★ Exposure별 가중치 매핑 (극저조도에서 Light Loss 감소) ★★★
        self.exposure_weights = {
            '001ms': {'light': 0.3, 'sfp': 1.5, 'use_ref': True},   # 극저조도: light↓, sfp↑, 200ms 참조
            '005ms': {'light': 0.5, 'sfp': 1.3, 'use_ref': True},
            '010ms': {'light': 0.7, 'sfp': 1.2, 'use_ref': True},
            '050ms': {'light': 0.9, 'sfp': 1.0, 'use_ref': False},  # 중간: 원본 사용
            '100ms': {'light': 1.0, 'sfp': 1.0, 'use_ref': False},
            '200ms': {'light': 1.0, 'sfp': 1.0, 'use_ref': False},
            '500ms': {'light': 1.0, 'sfp': 0.8, 'use_ref': False},  # 과노출: sfp 약간↓
        }
        
        # 손실 함수 인스턴스화
        self.loss_grad_consistency = GradientConsistencyLoss()
        self.loss_light = LightConsistencyLoss(**kwargs)
        self.loss_sfp = SpatialConsistencyLoss(device)
        self.loss_gamma_smooth = GammaSmoothnessLoss()
        self.loss_depth = DepthConsistencyLoss()
        self.loss_color = ColorConsistencyLoss()

    def _get_homography_from_calib(self, calib_data, device):
        """Calibration data에서 Homography 행렬 계산"""
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

    def _get_exposure_weights(self, exposure):
        """
        ★★★ 노출도에 따른 동적 가중치 반환 ★★★
        """
        if exposure is None:
            return {'light': 1.0, 'sfp': 1.0, 'use_ref': False}
        
        exp_str = exposure[0] if isinstance(exposure, list) else exposure
        if exp_str is None:
            return {'light': 1.0, 'sfp': 1.0, 'use_ref': False}
            
        # 정확한 매칭 시도
        if exp_str in self.exposure_weights:
            return self.exposure_weights[exp_str]
        
        # 부분 매칭 (예: '001ms' in '001ms_scene1')
        for key in self.exposure_weights:
            if key in str(exp_str):
                return self.exposure_weights[key]
        
        return {'light': 1.0, 'sfp': 1.0, 'use_ref': False}

    def forward(self, original_l, original_r, pred_l, pred_r, 
                fused_gamma_l, fused_gamma_r, depth_map_l, depth_map_r, calib_data, 
                exposure=None, ref_l=None, ref_r=None):
        """
        Args:
            original_l, original_r: 원본 이미지 (현재 노출)
            pred_l, pred_r: Enhanced 이미지
            fused_gamma_l, fused_gamma_r: Gamma maps
            depth_map_l, depth_map_r: Depth maps
            calib_data: Calibration 데이터
            exposure: 현재 노출 시간 (예: '001ms')
            ref_l, ref_r: ★★★ 200ms 참조 이미지 (극저조도에서 SFP 참조) ★★★
        """
        device = pred_l.device
        
        # ★★★ 노출도별 동적 가중치 획득 ★★★
        exp_weights = self._get_exposure_weights(exposure)
        light_weight_scale = exp_weights['light']
        sfp_weight_scale = exp_weights['sfp']
        use_reference = exp_weights['use_ref'] and self.use_reference_sfp
        
        # --- 1. Stereo Consistency Loss ---
        loss_s = torch.tensor(0.0, device=device)
        if self.lambda_stereo > 0:
            H_l_from_r = self._get_homography_from_calib(calib_data, device)
            H_l_from_r_b = H_l_from_r.unsqueeze(0).expand(pred_l.size(0), -1, -1)
            
            mask_r = torch.ones_like(pred_r)
            mask_warped = kornia.geometry.transform.warp_perspective(
                mask_r, H_l_from_r_b, (pred_l.shape[2], pred_l.shape[3]))
            mask_warped = (mask_warped > 0.99).float()

            pred_r_warped = kornia.geometry.transform.warp_perspective(
                pred_r, H_l_from_r_b, (pred_l.shape[2], pred_l.shape[3]))
            loss_s = self.loss_grad_consistency(pred_l * mask_warped, pred_r_warped * mask_warped)

        # --- 2. Depth Consistency Loss ---
        loss_d = torch.tensor(0.0, device=device)
        if self.lambda_depth > 0 and (depth_map_l is not None or depth_map_r is not None):
            exp_str = exposure[0] if isinstance(exposure, list) and len(exposure) > 0 else exposure
            loss_d = self.loss_depth(pred_l, pred_r, depth_map_l, depth_map_r, exposure=exp_str)

        # --- 3. Light Consistency Loss (★ 노출도별 동적 가중치) ---
        loss_light = self.loss_light(pred_l) + self.loss_light(pred_r)
        effective_w_light = self.w_light * light_weight_scale

        # --- 4. ★★★ Reference-based SFP Loss ★★★ ---
        # 극저조도: 200ms 참조 이미지와 비교 (구조 학습)
        # 중간 노출: 원본과 비교 (구조 보존)
        if use_reference and ref_l is not None and ref_r is not None:
            # 극저조도: "200ms처럼 보이게 향상하라"
            loss_sfp = (self.loss_sfp(ref_l, pred_l) + 
                       self.loss_sfp(ref_r, pred_r))
        else:
            # 중간/고노출: 원본 구조 보존
            loss_sfp = (self.loss_sfp(original_l, pred_l) + 
                       self.loss_sfp(original_r, pred_r))
        effective_w_sfp = self.w_sfp * sfp_weight_scale

        # --- 5. Gamma Smoothness Loss ---
        loss_gamma_smooth = (self.loss_gamma_smooth(fused_gamma_l) + 
                            self.loss_gamma_smooth(fused_gamma_r))

        # --- 6. Color Ratio Preservation Loss ---
        # ★★★ 극저조도에서는 색상 보존 약화 (원본 색상 정보 부족) ★★★
        loss_color = torch.tensor(0.0, device=device)
        if self.w_color > 0:
            if use_reference and ref_l is not None:
                # 극저조도: 참조 이미지의 색상 비율 따르기
                loss_color = (self.loss_color(ref_l, pred_l) + 
                             self.loss_color(ref_r, pred_r))
            else:
                loss_color = (self.loss_color(original_l, pred_l) + 
                             self.loss_color(original_r, pred_r))

        # --- 최종 손실 계산 (동적 가중치 적용) ---
        total_loss = (self.lambda_stereo * loss_s + 
                      self.lambda_depth * loss_d +
                      effective_w_light * loss_light +
                      effective_w_sfp * loss_sfp +
                      self.w_gamma * loss_gamma_smooth +
                      self.w_color * loss_color)
        
        loss_dict = {
            "total": total_loss.item(),
            "stereo": loss_s.item(),
            "depth": loss_d.item(),
            "light": loss_light.item(),
            "sfp": loss_sfp.item(),
            "gamma_smooth": loss_gamma_smooth.item(),
            "color": loss_color.item(),
            "light_w": effective_w_light,  # 디버깅용
            "sfp_w": effective_w_sfp,       # 디버깅용
        }
        
        return total_loss, loss_dict


# Alias for compatibility
ImprovedDimCamLoss = DimCamLoss
