# DimCam2/model.py (NAFBlock for Self-Refinement)

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. DPCE-Net 및 헬퍼 함수 임포트 ---
try:
    from DPCE2.model import enhance_net_nopool as DPCENet
    from DPCE2.model import gamma_enhance
    print("Successfully imported DPCE-Net from 'DPCE2' folder.")
except ImportError:
    print("ERROR: Could not import from 'DPCE2' folder.")
    raise


try:
    from local_arch import TiledInferenceWrapper
except ImportError:
    print("Could not import TiledInferenceWrapper from local_arch.py")
    # Tiled Inference 없이 작동하도록 임시 클래스 정의
    class TiledInferenceWrapper(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("Warning: TiledInferenceWrapper not found, running without tiled inference.")
        def forward(self, *args, **kwargs):
            return self.forward_core(*args, **kwargs)
# --- 2. ★★★ NAFBlock 및 관련 클래스 추가 ★★★ ---

class LayerNorm2d(nn.Module):
    """ 2D 특징맵을 위한 Layer Normalization (기존과 동일) """
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
    """ NAFNet에서 사용하는 SimpleGate """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """ NAFNet의 핵심 블록 """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

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


# model.py 파일에서 아래 SCAM 클래스로 교체

# model.py에 적용할 최종 SCAM 클래스

class SCAM(nn.Module):
    """
    Stereo Cross Attention Module (NAFSSR 기반 + A4 Windowed Attention)
    disparity_range 내에서만 attend하여 메모리 절감. 해상도 변화 없음.
    """
    def __init__(self, c, disparity_range=64):
        super().__init__()
        self.scale = c ** -0.5
        self.disparity_range = disparity_range
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)

        self.q_proj = nn.Conv2d(c, c, 1)
        self.k_proj = nn.Conv2d(c, c, 1)
        self.v_proj = nn.Conv2d(c, c, 1)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def _windowed_attention(self, q, k, v, W, d):
        """
        Windowed attention along width axis.
        q: [B,H,W,C], k: [B,H,C,W], v: [B,H,W,C]
        각 위치 w에서 [w-d, w+d] 범위의 key/value만 참조.
        """
        B, H, _, C = q.shape
        window = 2 * d + 1

        # k: [B,H,C,W] → k_padded: [B,H,C,W+2d]
        k_padded = F.pad(k, (d, d), mode='constant', value=0)
        # v: [B,H,W,C] → W축(dim=2) 패딩 → [B,H,W+2d,C]
        v_padded = F.pad(v, (0, 0, d, d), mode='constant', value=0)

        # Unfold to create windows
        # k_padded [B,H,C,W+2d] → unfold on dim=-1 → [B,H,C,W,window]
        k_windows = k_padded.unfold(-1, window, 1)  # [B,H,C,W,window]
        # v_padded [B,H,W+2d,C] → unfold on dim=2 → [B,H,W,window,C]
        v_windows = v_padded.unfold(2, window, 1)  # [B,H,W,window,C]

        # q: [B,H,W,C] → [B,H,W,1,C]
        q_expanded = q.unsqueeze(3)
        # k_windows: [B,H,C,W,window] → permute → [B,H,W,C,window]
        k_windows = k_windows.permute(0, 1, 3, 2, 4)

        # attn: [B,H,W,1,C] @ [B,H,W,C,window] → [B,H,W,1,window]
        attn = torch.matmul(q_expanded, k_windows) * self.scale

        # Mask for padded positions (edge handling)
        positions = torch.arange(W, device=q.device)
        # For each position w, valid range is [max(0,w-d), min(W-1,w+d)]
        window_positions = positions.unsqueeze(1) + torch.arange(-d, d+1, device=q.device).unsqueeze(0)  # [W, window]
        mask = (window_positions >= 0) & (window_positions < W)  # [W, window]
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1,1,W,1,window]
        attn = attn.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = attn.masked_fill(~mask, 0.0)  # NaN 방지

        # output: [B,H,W,1,window] @ [B,H,W,window,C] → [B,H,W,1,C] → [B,H,W,C]
        out = torch.matmul(attn, v_windows).squeeze(3)
        return out

    def forward(self, x_l, x_r):
        B, C, H, W = x_l.shape
        d = min(self.disparity_range, W // 2)  # W가 작으면 범위 제한

        # A2: norm 결과 캐싱
        x_l_norm = self.norm_l(x_l)
        x_r_norm = self.norm_r(x_r)

        q_l = self.q_proj(x_l_norm).permute(0, 2, 3, 1)  # [B,H,W,C]
        k_l_T = self.k_proj(x_l_norm).permute(0, 2, 1, 3)  # [B,H,C,W]
        v_l = self.v_proj(x_l).permute(0, 2, 3, 1)

        q_r = self.q_proj(x_r_norm).permute(0, 2, 3, 1)
        k_r_T = self.k_proj(x_r_norm).permute(0, 2, 1, 3)
        v_r = self.v_proj(x_r).permute(0, 2, 3, 1)

        # A4: Windowed Attention (디스패리티 범위 내만 attend)
        if d < W // 2:
            # Windowed mode — 메모리 절감
            F_r2l = self._windowed_attention(q_l, k_r_T, v_r, W, d)
            F_l2r = self._windowed_attention(q_r, k_l_T, v_l, W, d)
        else:
            # Full attention fallback (작은 이미지)
            attn_r2l = torch.matmul(q_l, k_r_T) * self.scale
            F_r2l = torch.matmul(torch.softmax(attn_r2l, dim=-1), v_r)
            attn_l2r = torch.matmul(q_r, k_l_T) * self.scale
            F_l2r = torch.matmul(torch.softmax(attn_l2r, dim=-1), v_l)

        delta_l = F_r2l.permute(0, 3, 1, 2) * self.beta
        delta_r = F_l2r.permute(0, 3, 1, 2) * self.gamma

        return delta_l, delta_r



class DepthNetWrapper(nn.Module):
    """ DepthNetWrapper (기존과 동일) """
    # ... (DepthNetWrapper 클래스 코드는 변경 없음) ...
    def __init__(self, model_type="MiDaS"):
        super().__init__()
        print(f"Initializing DepthNetWrapper with {model_type}...")
        self.proj = nn.Conv2d(6, 3, kernel_size=1, bias=True)
        try:
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        except Exception as e:
            print(f"Failed to load MiDaS from torch.hub: {e}")
            raise
        for param in self.depth_model.parameters():
            param.requires_grad = False
        print("Pretrained MiDaS model loaded and frozen.")
    def forward(self, x):
        x_proj = self.proj(x)
        self.depth_model.eval()
        with torch.no_grad():
            prediction = self.depth_model(x_proj)
            prediction = prediction.unsqueeze(1)
        return prediction


# --- 4. ★★★ 메인 모델: DimCamEnhancer가 TiledInferenceWrapper를 상속받도록 수정 ★★★ ---
class DimCamEnhancer(TiledInferenceWrapper):
    def __init__(self, use_tiled_inference=False, **kwargs): # ★★★ use_tiled_inference 인자 추가
        # Wrapper의 __init__을 먼저 호출하여 patch_size, overlap 설정
        wrapper_kwargs = {
            'patch_size': kwargs.get('patch_size', 128),
            'overlap': kwargs.get('overlap', 32)
        }
        super().__init__(**wrapper_kwargs)
        self.use_tiled_inference = use_tiled_inference


        # 모델의 핵심 로직 초기화
        _wrapper_keys = {'patch_size', 'overlap'}
        core_kwargs = {k: v for k, v in kwargs.items() if k not in _wrapper_keys}
        self._init_core_model(**core_kwargs)

    def _init_core_model(self, img_size=512, gamma_channels=3, img_channels=3,
                         embed_dim=48, num_blocks=4, lambda_depth=0.0,
                         use_grayscale=True, disparity_range=64):  # ★ grayscale/RGB 선택 옵션
        """ 모델의 실제 레이어들을 초기화하는 메소드 """
        self.use_grayscale = use_grayscale
        self.dce_net = DPCENet()
        self.intro = nn.Conv2d(img_channels * 2 + gamma_channels, embed_dim, 3, 1, 1)
        self.refine_blocks_l = nn.ModuleList([NAFBlock(embed_dim) for _ in range(num_blocks)])
        self.refine_blocks_r = nn.ModuleList([NAFBlock(embed_dim) for _ in range(num_blocks)])
        self.cross_attention = SCAM(embed_dim, disparity_range=disparity_range)
        self.outro = nn.Conv2d(embed_dim, gamma_channels, 3, 1, 1)
        
        self.lambda_depth = lambda_depth
        if self.lambda_depth > 0: self.depth_net = DepthNetWrapper()
        else: self.depth_net = None

        self.initialize_weights()

    # model.py -> DimCamEnhancer.initialize_weights

    def initialize_weights(self):
        """ 가중치 초기화 메소드 """
        # SCAM의 beta 파라미터를 0으로 초기화 (Identity Initialization)
        nn.init.constant_(self.cross_attention.beta, 0)
        nn.init.constant_(self.cross_attention.gamma, 0)
        
        # Outro Conv의 가중치와 편향을 0으로 초기화
        nn.init.constant_(self.outro.weight, 0)
        if self.outro.bias is not None:
            nn.init.constant_(self.outro.bias, 0)


    def forward_core(self, img_l, img_r):
        """
        모델의 핵심 연산 로직. TiledInferenceWrapper가 이 메소드를 호출합니다.
        """
        _, gamma_map_l_raw = self.dce_net(img_l)
        _, gamma_map_r_raw = self.dce_net(img_r)

        # ★★★ Grayscale/RGB 모드 선택 ★★★
        if self.use_grayscale:
            # Grayscale 모드: RGB 채널 평균으로 동일한 gamma 적용
            gamma_map_l = gamma_map_l_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            gamma_map_r = gamma_map_r_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        else:
            # RGB 모드: 채널별 독립적인 gamma 적용
            gamma_map_l = gamma_map_l_raw
            gamma_map_r = gamma_map_r_raw

        # DPCE 초기 향상 결과 (Stage 2에서 gradient flow 위해 no_grad 제거)
        dpce_only_enhanced_l = gamma_enhance(img_l, gamma_map_l)
        dpce_only_enhanced_r = gamma_enhance(img_r, gamma_map_r)

        # A1: Transformer 입력에 dpce_enhanced 추가 (9ch = img + dpce_enhanced + gamma_map)
        transformer_input_l = torch.cat([img_l, dpce_only_enhanced_l, gamma_map_l], dim=1)
        transformer_input_r = torch.cat([img_r, dpce_only_enhanced_r, gamma_map_r], dim=1)
        
        x_l, x_r = self.intro(transformer_input_l), self.intro(transformer_input_r)

        for block_l, block_r in zip(self.refine_blocks_l, self.refine_blocks_r):
            x_l = block_l(x_l)
            x_r = block_r(x_r)
        
        x_l_sa, x_r_sa = x_l, x_r

        delta_l, delta_r = self.cross_attention(x_l_sa, x_r_sa)

        
        x_l_final, x_r_final = x_l_sa + delta_l, x_r_sa + delta_r
        
        # Outro: gamma delta 계산
        gamma_l_delta_raw = self.outro(x_l_final)
        gamma_r_delta_raw = self.outro(x_r_final)

        # ★★★ Grayscale/RGB 모드에 따라 처리 ★★★
        if self.use_grayscale:
            # Grayscale 모드: 채널 평균으로 동일한 delta 적용
            gamma_l_delta = gamma_l_delta_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            gamma_r_delta = gamma_r_delta_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        else:
            # RGB 모드: 채널별 독립적인 delta 적용
            gamma_l_delta = gamma_l_delta_raw
            gamma_r_delta = gamma_r_delta_raw

        # A3: Illumination-aware delta — 어두운 영역에 더 큰 보정
        illum_l = img_l.mean(dim=1, keepdim=True)
        illum_r = img_r.mean(dim=1, keepdim=True)
        dark_mask_l = (1.0 - illum_l).clamp(0, 1)
        dark_mask_r = (1.0 - illum_r).clamp(0, 1)

        fused_gamma_l = gamma_map_l + gamma_l_delta * dark_mask_l
        fused_gamma_r = gamma_map_r + gamma_r_delta * dark_mask_r
        
        epsilon = 1e-6
        enhanced_l = gamma_enhance(img_l, fused_gamma_l.clamp(min=epsilon))
        enhanced_r = gamma_enhance(img_r, fused_gamma_r.clamp(min=epsilon))

        depth_map = self.depth_net(torch.cat([enhanced_l, enhanced_r], dim=1)) if self.depth_net else None

        return (enhanced_l, enhanced_r, depth_map, dpce_only_enhanced_l, 
                dpce_only_enhanced_r, fused_gamma_l, fused_gamma_r)

    def forward(self, img_l, img_r):
        # ★★★ Tiled Inference 사용 여부를 명시적으로 제어 ★★★
        if self.use_tiled_inference and not self.training:
            # use_tiled_inference가 True이고, 추론 모드일 때만 Wrapper 호출
            return TiledInferenceWrapper.forward(self, img_l, img_r)
        else:
            # 그 외의 모든 경우 (학습, 검증)에는 forward_core를 직접 호출
            return self.forward_core(img_l, img_r)