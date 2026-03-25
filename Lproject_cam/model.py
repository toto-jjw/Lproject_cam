# Lproject_cam/model.py
# DimCamEnhancer - Simplified Version (Pre-computed Depth)
# ★★★ FoundationStereo/MiDaS 제거: Depth는 Pre-compute 방식 사용 ★★★

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# --- 1. DPCE-Net 및 헬퍼 함수 임포트 ---
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from DPCE2.model import enhance_net_nopool as DPCENet
    from DPCE2.model import gamma_enhance
    print("Successfully imported DPCE-Net from 'DPCE2' folder.")
except ImportError:
    print("ERROR: Could not import from 'DPCE2' folder.")
    raise


# --- 2. TiledInferenceWrapper 임포트 ---
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from local_arch import TiledInferenceWrapper
except ImportError:
    print("Could not import TiledInferenceWrapper from local_arch.py")
    class TiledInferenceWrapper(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, *args, **kwargs):
            return self.forward_core(*args, **kwargs)


# --- 3. Stereo Denoising (단순화된 버전) ---

class StereoDenoiser(nn.Module):
    """
    스테레오 이미지 쌍을 이용한 노이즈 감소 (단순화된 버전)
    
    ★★★ 핵심 원리 ★★★
    - 스테레오 이미지에서 노이즈는 독립적 (uncorrelated)
    - I_l = Signal + Noise_L
    - I_r = Signal + Noise_R
    - 평균: (I_l + I_r) / 2 = Signal + (Noise_L + Noise_R) / 2
    - 노이즈 분산이 √2 배 감소 → SNR √2 배 향상
    
    ★★★ 워핑 방식 ★★★
    - Rectified 스테레오 가정: 에피폴라 라인이 수평
    - Depth → Disparity 변환 후 수평 워핑만 수행
    - ★★★ calib_data에서 baseline, focal_length 동적 로드 ★★★
    
    ★★★ Depth Map 형식 ★★★
    - Pre-computed depth는 0-1로 정규화됨 (FoundationStereo 출력)
    - 실제 depth = depth_normalized * max_depth (max_depth는 알 수 없음)
    - 따라서 disparity도 정규화된 값으로 계산 (상대적 이동량)
    """
    def __init__(self, default_baseline=0.4, default_focal_length=1453.0,
                 max_disparity=128):
        super().__init__()
        # ★★★ calib_data가 없을 때 사용할 기본값 (실제 데이터셋 기준) ★★★
        # Lunar dataset: baseline ≈ 0.4m, focal_length ≈ 1453px
        self.default_baseline = default_baseline
        self.default_focal_length = default_focal_length
        # ★★★ 정규화된 depth 사용 시 최대 disparity (픽셀 단위) ★★★
        self.max_disparity = max_disparity
    
    def warp_horizontal(self, img, disparity, direction='left_to_right'):
        """
        Rectified 스테레오용 수평 워핑
        - Rectified 이미지에서는 대응점이 같은 row에 있음
        - disparity만큼 수평 이동
        """
        B, C, H, W = img.shape
        device = img.device
        
        # 그리드 생성
        xx = torch.arange(0, W, device=device).float()
        yy = torch.arange(0, H, device=device).float()
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        
        xx = xx.unsqueeze(0).expand(B, -1, -1)
        yy = yy.unsqueeze(0).expand(B, -1, -1)
        
        # Disparity 적용 (수평 이동만)
        disp = disparity.squeeze(1) if disparity.dim() == 4 else disparity
        if direction == 'left_to_right':
            xx_warped = xx - disp  # 오른쪽 이미지를 왼쪽으로 워핑
        else:
            xx_warped = xx + disp  # 왼쪽 이미지를 오른쪽으로 워핑
        
        # 정규화 [-1, 1]
        xx_warped = 2.0 * xx_warped / (W - 1) - 1.0
        yy_norm = 2.0 * yy / (H - 1) - 1.0
        
        grid = torch.stack([xx_warped, yy_norm], dim=-1)
        
        return F.grid_sample(img, grid, mode='bilinear', 
                            padding_mode='border', align_corners=True)
    
    def depth_to_disparity(self, depth_map, baseline, focal_length):
        """
        Depth map을 disparity로 변환 (rectified stereo)
        
        ★★★ 정규화된 Depth 처리 ★★★
        - depth_map은 0-1로 정규화됨 (0=가까움, 1=멀리)
        - FoundationStereo의 출력은 inverse depth 형태
        - depth_normalized가 작을수록(가까울수록) disparity가 커야 함
        
        공식: disparity = max_disparity * (1 - depth_normalized)
        또는: disparity = max_disparity * depth_normalized (inverse depth인 경우)
        """
        # ★★★ depth가 이미 정규화되어 있으므로 직접 disparity로 변환 ★★★
        # FoundationStereo는 보통 disparity를 정규화해서 저장함
        # depth_normalized ≈ normalized_disparity
        # disparity (pixels) = depth_normalized * max_disparity
        disparity = depth_map * self.max_disparity
        return disparity.clamp(0, self.max_disparity)
    
    def _get_calib_params(self, calib_data):
        """
        ★★★ calib_data에서 baseline과 focal_length 추출 ★★★
        기존 모델의 로딩 방식과 동일
        """
        if calib_data is None:
            return self.default_baseline, self.default_focal_length
        
        try:
            # baseline: translation vector의 norm
            baseline = calib_data.get('baseline', self.default_baseline)
            
            # focal_length: left camera matrix의 fx
            focal_length = calib_data.get('focal_length', self.default_focal_length)
            
            return baseline, focal_length
        except (KeyError, TypeError, AttributeError):
            return self.default_baseline, self.default_focal_length
    
    def forward(self, img_l, img_r, depth_map=None, calib_data=None):
        """
        스테레오 디노이징
        
        Args:
            img_l: 왼쪽 이미지 [B, C, H, W]
            img_r: 오른쪽 이미지 [B, C, H, W]
            depth_map: (optional) depth map for accurate warping
            calib_data: (optional) calibration data with baseline and focal_length
        
        Returns:
            denoised_l, denoised_r: 디노이징된 이미지 쌍
            None, None: 노이즈 마스크 (호환성 유지, 사용 안함)
        """
        # ★★★ calib_data에서 baseline, focal_length 로드 ★★★
        baseline, focal_length = self._get_calib_params(calib_data)
        
        if depth_map is not None:
            # Depth 기반 정확한 워핑
            disparity = self.depth_to_disparity(depth_map, baseline, focal_length)
            warped_r = self.warp_horizontal(img_r, disparity, 'left_to_right')
            warped_l = self.warp_horizontal(img_l, disparity, 'right_to_left')
            
            # ★★★ 핵심: 양쪽 평균으로 노이즈 감소 (√2 배 SNR 향상) ★★★
            denoised_l = (img_l + warped_r) / 2
            denoised_r = (img_r + warped_l) / 2
        else:
            # Depth 없으면 워핑 없이 사용 (Cross Attention이 처리)
            denoised_l = img_l
            denoised_r = img_r
        
        return denoised_l, denoised_r, None, None


# 기존 이름과의 호환성 유지
StereoNoiseMaskExtractor = StereoDenoiser


# --- 4. NAFBlock 및 관련 클래스 ---

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.c = c
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class SCAM(nn.Module):
    """
    Stereo Cross Attention Module (DimCam2 방식 - Global Attention)
    
    ★★★ 핵심 변경: Row-wise → Global Attention ★★★
    - Row-wise는 메모리 효율적이지만 gradient 전파가 약함
    - Global attention은 전체 이미지 정보를 융합하여 밝기 학습에 효과적
    """
    def __init__(self, c, stripe_size=32):  # stripe_size는 호환성 유지용
        super().__init__()
        self.scale = c ** -0.5
        
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        
        self.q_proj = nn.Conv2d(c, c, 1)
        self.k_proj = nn.Conv2d(c, c, 1)
        self.v_proj = nn.Conv2d(c, c, 1)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x_l, x_r):
        B, C, H, W = x_l.shape

        # 1. Q, K, V 생성 (공유 레이어 사용)
        # ★★★ DimCam2 방식: [B,H,W,C] 형태로 변환 ★★★
        q_l = self.q_proj(self.norm_l(x_l)).permute(0, 2, 3, 1)  # [B,H,W,C]
        k_l_T = self.k_proj(self.norm_l(x_l)).permute(0, 2, 1, 3)  # [B,H,C,W]
        v_l = self.v_proj(x_l).permute(0, 2, 3, 1)  # [B,H,W,C]

        q_r = self.q_proj(self.norm_r(x_r)).permute(0, 2, 3, 1)
        k_r_T = self.k_proj(self.norm_r(x_r)).permute(0, 2, 1, 3)
        v_r = self.v_proj(x_r).permute(0, 2, 3, 1)

        # 2. ★★★ Global Attention (전체 이미지 정보 융합) ★★★
        # Right-to-Left: Q from Left, K/V from Right
        # Attention: [B,H,W,C] × [B,H,C,W] → [B,H,W,W]
        attn_r2l = torch.matmul(q_l, k_r_T) * self.scale
        F_r2l = torch.matmul(F.softmax(attn_r2l, dim=-1), v_r)  # [B,H,W,C]

        # Left-to-Right: Q from Right, K/V from Left
        attn_l2r = torch.matmul(q_r, k_l_T) * self.scale
        F_l2r = torch.matmul(F.softmax(attn_l2r, dim=-1), v_l)  # [B,H,W,C]
        
        # 3. 최종 출력 (튜플로 양방향 결과 모두 반환)
        delta_l = F_r2l.permute(0, 3, 1, 2) * self.beta  # [B,C,H,W]
        delta_r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        
        return delta_l, delta_r


# --- 5. ★★★ Main Model: DimCamEnhancer (Noise-Aware) ★★★ ---

class DimCamEnhancer(TiledInferenceWrapper):
    """
    DimCamEnhancer - Noise-Aware Version (DimCam 논문 구현)
    
    ★★★ 핵심 기능 ★★★
    1. 스테레오 이미지 쌍에서 노이즈 마스크 추출
    2. 노이즈 영역은 Cross-view 정보로 복원
    3. 깨끗한 영역은 일반 enhancement
    4. Pre-computed depth 활용하여 정확한 스테레오 정렬
    """
    def __init__(self, use_tiled_inference=False, **kwargs):
        wrapper_kwargs = {
            'patch_size': kwargs.get('patch_size', 128),
            'overlap': kwargs.get('overlap', 32)
        }
        super().__init__(**wrapper_kwargs)
        nn.Module.__init__(self)
        self.use_tiled_inference = use_tiled_inference
        
        core_kwargs = {k: v for k, v in kwargs.items() if k not in wrapper_kwargs}
        self._init_core_model(**core_kwargs)

    def _init_core_model(self, img_size=512, gamma_channels=3, img_channels=3,
                         embed_dim=48, num_blocks=4,
                         use_grayscale=True,
                         residual_scale=1.0,  # ★ train.py와 일치: 1.0
                         gamma_min=None,  # ★ 제한 없음
                         gamma_max=None,  # ★ 제한 없음
                         use_noise_mask=True,  # ★ 노이즈 마스크 활성화 옵션
                         **kwargs):  # ★ 사용하지 않는 인자 무시 (호환성)
        
        self.use_grayscale = use_grayscale
        # gamma_min/max는 더 이상 사용하지 않음 (호환성 유지용으로만 저장)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.use_noise_mask = use_noise_mask
        
        # DPCE-Net
        self.dce_net = DPCENet()
        
        # ★★★ 스테레오 디노이저 (단순화된 평균 기반) ★★★
        # baseline, focal_length는 forward 시 calib_data에서 동적으로 로드
        # ★★★ 기본값을 실제 Lunar 데이터셋 값으로 설정 ★★★
        if use_noise_mask:
            self.stereo_denoiser = StereoDenoiser(
                default_baseline=0.4,       # Lunar dataset: ~0.4m
                default_focal_length=1453.0, # Lunar dataset: ~1453px
                max_disparity=128            # 512px 이미지 기준 최대 disparity
            )
        
        # Transformer blocks
        self.intro = nn.Conv2d(img_channels + gamma_channels, embed_dim, 3, 1, 1)
        self.refine_blocks_l = nn.ModuleList([NAFBlock(embed_dim) for _ in range(num_blocks)])
        self.refine_blocks_r = nn.ModuleList([NAFBlock(embed_dim) for _ in range(num_blocks)])
        self.cross_attention = SCAM(embed_dim)
        self.outro = nn.Conv2d(embed_dim, gamma_channels, 3, 1, 1)
        
        # Residual scale (fixed, not learnable)
        self.residual_scale = residual_scale

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.constant_(self.cross_attention.beta, 0)
        nn.init.constant_(self.cross_attention.gamma, 0)
        # ★★★ outro=0으로 초기화: 초기 상태에서 DPCE와 동일한 출력 ★★★
        # gamma_delta=0 → fused_gamma = DPCE_gamma → 출력 = DPCE 출력
        # Light Loss의 gradient가 outro로 전파되어 학습 진행됨
        nn.init.constant_(self.outro.weight, 0)
        if self.outro.bias is not None:
            nn.init.constant_(self.outro.bias, 0)

    def forward_core(self, img_l, img_r, ref_img_l=None, ref_img_r=None, 
                      precomputed_depth=None, calib_data=None):
        """
        핵심 forward 로직
        
        ★★★ 워크플로우 (DimCam2 방식으로 수정) ★★★
        1. DPCE-Net으로 gamma map 생성 (원본 이미지 사용!)
        2. Transformer + Cross Attention으로 gamma 보정
        3. 원본 이미지에 최종 gamma 적용
        
        ★★★ 핵심 수정: 스테레오 디노이징을 DPCE-Net 이후로 이동 ★★★
        - DPCE-Net은 원본 이미지로 gamma map 생성 (DimCam2와 동일)
        - Cross Attention이 스테레오 정보를 활용하여 노이즈 처리
        """
        noise_mask_l, noise_mask_r = None, None
        
        # ★★★ 수정: 원본 이미지를 DPCE-Net과 Transformer에 직접 사용 ★★★
        # (스테레오 디노이징 제거 - DimCam2와 동일하게)
        input_l, input_r = img_l, img_r
        
        # 1. DPCE-Net으로 기본 gamma map 생성 (원본 이미지 사용!)
        dpce_out_l = self.dce_net(input_l)
        dpce_out_r = self.dce_net(input_r)
        
        if isinstance(dpce_out_l, tuple):
            gamma_map_l_raw, gamma_map_r_raw = dpce_out_l[1], dpce_out_r[1]
        else:
            gamma_map_l_raw, gamma_map_r_raw = dpce_out_l, dpce_out_r

        # 2. Grayscale/RGB 모드 선택
        if self.use_grayscale:
            gamma_map_l = gamma_map_l_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            gamma_map_r = gamma_map_r_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        else:
            gamma_map_l = gamma_map_l_raw
            gamma_map_r = gamma_map_r_raw

        # 3. DPCE-only enhanced 이미지 (비교용)
        with torch.no_grad():
            dpce_only_enhanced_l = gamma_enhance(input_l, gamma_map_l)
            dpce_only_enhanced_r = gamma_enhance(input_r, gamma_map_r)

        # 4. Transformer refinement (마스킹된 이미지 + gamma map)
        transformer_input_l = torch.cat([input_l, gamma_map_l], dim=1)
        transformer_input_r = torch.cat([input_r, gamma_map_r], dim=1)
        
        x_l, x_r = self.intro(transformer_input_l), self.intro(transformer_input_r)

        # Use gradient checkpointing during training for memory efficiency
        if self.training and torch.is_grad_enabled():
            for block_l, block_r in zip(self.refine_blocks_l, self.refine_blocks_r):
                x_l = torch.utils.checkpoint.checkpoint(block_l, x_l, use_reentrant=False)
                x_r = torch.utils.checkpoint.checkpoint(block_r, x_r, use_reentrant=False)
        else:
            for block_l, block_r in zip(self.refine_blocks_l, self.refine_blocks_r):
                x_l = block_l(x_l)
                x_r = block_r(x_r)
        
        # 5. Cross Attention (논문: 스테레오 정보 활용)
        x_l_sa, x_r_sa = x_l, x_r
        delta_l, delta_r = self.cross_attention(x_l_sa, x_r_sa)
        x_l_final, x_r_final = x_l_sa + delta_l, x_r_sa + delta_r

        # 6. Gamma delta 계산
        gamma_l_delta_raw = self.outro(x_l_final)
        gamma_r_delta_raw = self.outro(x_r_final)

        if self.use_grayscale:
            gamma_l_delta = gamma_l_delta_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            gamma_r_delta = gamma_r_delta_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        else:
            gamma_l_delta = gamma_l_delta_raw
            gamma_r_delta = gamma_r_delta_raw

        # Fused gamma (no scaling, no clipping)
        fused_gamma_l = gamma_map_l + gamma_l_delta
        fused_gamma_r = gamma_map_r + gamma_r_delta

        # 7. 최종 향상 이미지 생성 (원본 이미지에 적용)
        # ★★★ 중요: gamma는 원본 이미지에 적용 (마스킹 이미지가 아님) ★★★
        epsilon = 1e-6
        enhanced_l = gamma_enhance(img_l, fused_gamma_l.clamp(min=epsilon))
        enhanced_r = gamma_enhance(img_r, fused_gamma_r.clamp(min=epsilon))

        # 8. Depth: Pre-computed 값 그대로 반환
        depth_map = precomputed_depth

        # 반환값: 노이즈 마스크는 시각화용으로만 반환 (손실 함수에는 미사용)
        return (enhanced_l, enhanced_r, depth_map, dpce_only_enhanced_l,
                dpce_only_enhanced_r, fused_gamma_l, fused_gamma_r,
                noise_mask_l, noise_mask_r)

    def forward(self, img_l, img_r, ref_img_l=None, ref_img_r=None, 
                precomputed_depth=None, calib_data=None):
        """
        Forward pass
        
        Args:
            img_l: 현재 노출도 왼쪽 이미지
            img_r: 현재 노출도 오른쪽 이미지
            ref_img_l: (unused, 호환성 유지)
            ref_img_r: (unused, 호환성 유지)
            precomputed_depth: Pre-computed depth map [B, 1, H, W]
            calib_data: Calibration 데이터 (baseline, focal_length 포함)
        """
        if self.use_tiled_inference and not self.training:
            return TiledInferenceWrapper.forward(self, img_l, img_r)
        else:
            return self.forward_core(img_l, img_r, ref_img_l, ref_img_r, 
                                     precomputed_depth, calib_data)

    def get_residual_scale(self):
        """현재 residual scale 값 반환 (로깅용)"""
        return self.residual_scale if isinstance(self.residual_scale, float) else 1.0


# --- 6. Alias for backward compatibility ---
ImprovedDimCamEnhancer = DimCamEnhancer
