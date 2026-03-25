# DimCam2/model.py (NAFBlock for Self-Refinement)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


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
    Stereo Cross Attention Module (NAFSSR의 병렬/양방향 방식 적용)
    한 번의 forward 호출로 양방향 어텐션 결과를 모두 계산합니다.
    """
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        
        # 양방향 계산에 필요한 모든 프로젝션 레이어 정의
        self.q_proj = nn.Conv2d(c, c, 1) # Q는 l, r 모두에서 생성되므로 공유
        self.k_proj = nn.Conv2d(c, c, 1) # K도 공유
        self.v_proj = nn.Conv2d(c, c, 1) # V도 공유
        
        # 양방향 결과에 각각 적용될 파라미터
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True) # for Right-to-Left
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True) # for Left-to-Right

    def forward(self, x_l, x_r):
        B, C, H, W = x_l.shape

        # 1. Q, K, V 생성 (공유 레이어 사용)
        q_l = self.q_proj(self.norm_l(x_l)).permute(0, 2, 3, 1) # [B,H,W,C]
        k_l_T = self.k_proj(self.norm_l(x_l)).permute(0, 2, 1, 3) # [B,H,C,W]
        v_l = self.v_proj(x_l).permute(0, 2, 3, 1) # [B,H,W,C]

        q_r = self.q_proj(self.norm_r(x_r)).permute(0, 2, 3, 1)
        k_r_T = self.k_proj(self.norm_r(x_r)).permute(0, 2, 1, 3)
        v_r = self.v_proj(x_r).permute(0, 2, 3, 1)

        # 2. 병렬 어텐션 계산
        # Right-to-Left: Q from Left, K/V from Right
        attn_r2l = torch.matmul(q_l, k_r_T) * self.scale
        F_r2l = torch.matmul(torch.softmax(attn_r2l, dim=-1), v_r) # [B,H,W,C]

        # Left-to-Right: Q from Right, K/V from Left
        attn_l2r = torch.matmul(q_r, k_l_T) * self.scale
        F_l2r = torch.matmul(torch.softmax(attn_l2r, dim=-1), v_l) # [B,H,W,C]
        
        # 3. 최종 출력 (튜플로 양방향 결과 모두 반환)
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
        
        # nn.Module의 __init__을 명시적으로 호출
        nn.Module.__init__(self)
        self.use_tiled_inference = use_tiled_inference


        # 모델의 핵심 로직 초기화
        # kwargs에서 wrapper 관련 인자를 제거하고 전달
        core_kwargs = {k: v for k, v in kwargs.items() if k not in wrapper_kwargs}
        self._init_core_model(**core_kwargs)

    def _init_core_model(self, img_size=512, gamma_channels=3, img_channels=3,
                         embed_dim=48, num_blocks=4, lambda_depth=0.0,
                         use_grayscale=True):  # ★ grayscale/RGB 선택 옵션
        """ 모델의 실제 레이어들을 초기화하는 메소드 """
        self.use_grayscale = use_grayscale
        self.dce_net = DPCENet()
        self.intro = nn.Conv2d(img_channels + gamma_channels, embed_dim, 3, 1, 1)
        self.refine_blocks_l = nn.ModuleList([NAFBlock(embed_dim) for _ in range(num_blocks)])
        self.refine_blocks_r = nn.ModuleList([NAFBlock(embed_dim) for _ in range(num_blocks)])
        self.cross_attention = SCAM(embed_dim)
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
        if isinstance(self.dce_net(img_l), tuple):
            gamma_map_l_raw, gamma_map_r_raw = self.dce_net(img_l)[1], self.dce_net(img_r)[1]
        else:
            gamma_map_l_raw, gamma_map_r_raw = self.dce_net(img_l), self.dce_net(img_r)

        # ★★★ Grayscale/RGB 모드 선택 ★★★
        if self.use_grayscale:
            # Grayscale 모드: RGB 채널 평균으로 동일한 gamma 적용
            gamma_map_l = gamma_map_l_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            gamma_map_r = gamma_map_r_raw.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        else:
            # RGB 모드: 채널별 독립적인 gamma 적용
            gamma_map_l = gamma_map_l_raw
            gamma_map_r = gamma_map_r_raw

        with torch.no_grad():
            dpce_only_enhanced_l = gamma_enhance(img_l, gamma_map_l)
            dpce_only_enhanced_r = gamma_enhance(img_r, gamma_map_r)

        
            
        transformer_input_l = torch.cat([img_l, gamma_map_l], dim=1)
        transformer_input_r = torch.cat([img_r, gamma_map_r], dim=1)
        
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

        
        fused_gamma_l = gamma_map_l + gamma_l_delta
        fused_gamma_r = gamma_map_r + gamma_r_delta
        
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