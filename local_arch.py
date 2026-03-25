# local_arch.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TiledInferenceWrapper(nn.Module):
    """
    Tiled Inference 기능을 제공하는 기반 클래스.
    이 클래스를 상속받는 모델은 큰 이미지를 패치 단위로 처리할 수 있습니다.
    """
    def __init__(self, patch_size=128, overlap=32):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        print(f"TiledInferenceWrapper initialized with patch_size={patch_size}, overlap={overlap}")

    def forward(self, *args, **kwargs):
        """
        입력 이미지가 패치 크기보다 크면 Tiled Inference를 수행하고,
        그렇지 않으면 (또는 학습 중이면) 원본 forward_core를 호출합니다.
        """
        # 모델의 핵심 로직은 forward_core에 구현되어 있어야 합니다.
        if not hasattr(self, 'forward_core'):
            raise NotImplementedError("Models inheriting from TiledInferenceWrapper must implement a 'forward_core' method.")

        # 학습 모드이거나, 입력 이미지가 패치 크기보다 작으면 바로 처리
        if self.training or (args[0].shape[2] <= self.patch_size and args[0].shape[3] <= self.patch_size):
            return self.forward_core(*args, **kwargs)

        # --- 추론 모드에서 Tiled Inference 실행 ---
        # 스테레오 입력을 가정 (img_l, img_r)
        img_l, img_r = args[0], args[1]
        b, c, h, w = img_l.shape
        stride = self.patch_size - self.overlap
        
        result_l = torch.zeros_like(img_l)
        result_r = torch.zeros_like(img_r)
        weight_map = torch.zeros((b, 1, h, w), device=img_l.device)

        # 블렌딩을 위한 가중치 마스크 (Hanning Window)
        patch_weight = torch.hann_window(self.patch_size, periodic=False).unsqueeze(1) * \
                       torch.hann_window(self.patch_size, periodic=False).unsqueeze(0)
        patch_weight = patch_weight.unsqueeze(0).unsqueeze(0).to(img_l.device)

        y_coords = [i for i in range(0, h, stride) if i + self.patch_size <= h]
        if h % stride != 0: y_coords.append(h - self.patch_size)
            
        x_coords = [i for i in range(0, w, stride) if i + self.patch_size <= w]
        if w % stride != 0: x_coords.append(w - self.patch_size)

        for y in set(y_coords):
            for x in set(x_coords):
                patch_l = img_l[..., y:y+self.patch_size, x:x+self.patch_size]
                patch_r = img_r[..., y:y+self.patch_size, x:x+self.patch_size]
                
                # 모델의 핵심 로직 호출하여 패치 추론
                # 추론 시에는 최종 향상된 이미지 2개만 필요함
                enhanced_patch_l, enhanced_patch_r, *_ = self.forward_core(patch_l, patch_r)
                
                result_l[..., y:y+self.patch_size, x:x+self.patch_size] += enhanced_patch_l * patch_weight
                result_r[..., y:y+self.patch_size, x:x+self.patch_size] += enhanced_patch_r * patch_weight
                weight_map[..., y:y+self.patch_size, x:x+self.patch_size] += patch_weight
        
        final_l = result_l / (weight_map + 1e-8)
        final_r = result_r / (weight_map + 1e-8)

        return final_l, final_r

