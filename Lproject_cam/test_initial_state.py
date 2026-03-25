# test_initial_state.py
# 초기 상태 모델 출력 비교 테스트

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model
import dataloader

def test_initial_state():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 데이터 로드 ---
    data_path = os.path.expanduser('~/Downloads/lunardataset2/')
    dataset = dataloader.LunarStereoDataset(
        data_path, mode='val', transform=True, 
        img_height=512, img_width=512,
        use_all_exposures=True,
        reference_exposure='200ms',
        use_precomputed_depth=True
    )
    
    # 랜덤 인덱스 선택
    np.random.seed(42)
    indices = np.random.choice(len(dataset), 3, replace=False)
    
    # --- 모델 초기화 (outro=0) ---
    print("\n" + "="*60)
    print("Model 1: outro=0 초기화 (원래 DimCam2 방식)")
    print("="*60)
    
    dimcam_model_zero = model.DimCamEnhancer(
        img_size=512,
        embed_dim=64,
        num_blocks=5,
        use_grayscale=True,
        residual_scale=1.0,
        gamma_min=0.01,
        gamma_max=10.0,
        use_noise_mask=False,  # 노이즈 마스크 비활성화
    ).to(device)
    
    # outro를 0으로 재초기화
    nn.init.constant_(dimcam_model_zero.outro.weight, 0)
    nn.init.constant_(dimcam_model_zero.outro.bias, 0)
    print("outro weights: all zeros")
    print(f"outro.weight.abs().sum() = {dimcam_model_zero.outro.weight.abs().sum().item()}")
    
    # DPCE 가중치 로드
    dpce_path = '../DPCE2/snapshots/Epoch_200_original.pth'
    if os.path.exists(dpce_path):
        dimcam_model_zero.dce_net.load_state_dict(torch.load(dpce_path, map_location=device))
        print("DPCE weights loaded successfully")
    
    # Freeze DPCE
    for param in dimcam_model_zero.dce_net.parameters():
        param.requires_grad = False
    
    dimcam_model_zero.eval()
    
    # --- 모델 2: outro=xavier 초기화 ---
    print("\n" + "="*60)
    print("Model 2: outro=xavier 초기화 (수정된 방식)")
    print("="*60)
    
    dimcam_model_xavier = model.DimCamEnhancer(
        img_size=512,
        embed_dim=64,
        num_blocks=5,
        use_grayscale=True,
        residual_scale=1.0,
        gamma_min=0.01,
        gamma_max=10.0,
        use_noise_mask=False,
    ).to(device)
    
    # outro는 이미 xavier로 초기화됨 (model.py의 initialize_weights)
    print(f"outro.weight.abs().sum() = {dimcam_model_xavier.outro.weight.abs().sum().item()}")
    
    if os.path.exists(dpce_path):
        dimcam_model_xavier.dce_net.load_state_dict(torch.load(dpce_path, map_location=device))
    
    for param in dimcam_model_xavier.dce_net.parameters():
        param.requires_grad = False
    
    dimcam_model_xavier.eval()
    
    # --- 테스트 실행 ---
    fig, axes = plt.subplots(len(indices), 5, figsize=(20, 4*len(indices)))
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            img_l, img_r, ref_l, ref_r, depth_l, depth_r, calib, exposure = sample
            
            img_l = img_l.unsqueeze(0).to(device)
            img_r = img_r.unsqueeze(0).to(device)
            depth_l = depth_l.unsqueeze(0).to(device) if depth_l is not None else None
            
            print(f"\n--- Sample {idx} (exposure: {exposure}) ---")
            print(f"Input mean brightness: {img_l.mean().item():.4f}")
            
            # Model 1: outro=0
            outputs_zero = dimcam_model_zero(img_l, img_r, None, None, depth_l)
            pred_zero = outputs_zero[0]
            dpce_only = outputs_zero[3]
            gamma_zero = outputs_zero[5]
            
            # Model 2: outro=xavier
            outputs_xavier = dimcam_model_xavier(img_l, img_r, None, None, depth_l)
            pred_xavier = outputs_xavier[0]
            gamma_xavier = outputs_xavier[5]
            
            print(f"[outro=0] gamma mean: {gamma_zero.mean().item():.4f}, output mean: {pred_zero.mean().item():.4f}")
            print(f"[outro=xavier] gamma mean: {gamma_xavier.mean().item():.4f}, output mean: {pred_xavier.mean().item():.4f}")
            print(f"[DPCE only] output mean: {dpce_only.mean().item():.4f}")
            
            # 출력 차이 분석
            diff_zero_dpce = (pred_zero - dpce_only).abs().mean().item()
            diff_xavier_dpce = (pred_xavier - dpce_only).abs().mean().item()
            diff_zero_orig = (pred_zero - img_l).abs().mean().item()
            diff_xavier_orig = (pred_xavier - img_l).abs().mean().item()
            
            print(f"[outro=0] vs DPCE: {diff_zero_dpce:.6f}")
            print(f"[outro=xavier] vs DPCE: {diff_xavier_dpce:.6f}")
            print(f"[outro=0] vs Original: {diff_zero_orig:.6f}")
            print(f"[outro=xavier] vs Original: {diff_xavier_orig:.6f}")
            
            # 시각화
            def to_np(t):
                return t[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            
            axes[row, 0].imshow(to_np(img_l))
            axes[row, 0].set_title(f'Original\nmean={img_l.mean().item():.3f}')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(to_np(dpce_only))
            axes[row, 1].set_title(f'DPCE Only\nmean={dpce_only.mean().item():.3f}')
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(to_np(pred_zero))
            axes[row, 2].set_title(f'outro=0\nmean={pred_zero.mean().item():.3f}')
            axes[row, 2].axis('off')
            
            axes[row, 3].imshow(to_np(pred_xavier))
            axes[row, 3].set_title(f'outro=xavier\nmean={pred_xavier.mean().item():.3f}')
            axes[row, 3].axis('off')
            
            # Gamma map 시각화
            gamma_vis = gamma_zero[0, 0].cpu().numpy()
            im = axes[row, 4].imshow(gamma_vis, cmap='viridis')
            axes[row, 4].set_title(f'Gamma (outro=0)\nrange=[{gamma_vis.min():.2f}, {gamma_vis.max():.2f}]')
            axes[row, 4].axis('off')
            plt.colorbar(im, ax=axes[row, 4], fraction=0.046)
            
            # 행 라벨
            axes[row, 0].set_ylabel(f'Exp: {exposure}', fontsize=12, rotation=0, labelpad=50)
    
    plt.suptitle('Initial Model State Comparison (No Training)', fontsize=14)
    plt.tight_layout()
    plt.savefig('initial_state_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Saved to 'initial_state_comparison.png'")
    
    # --- 핵심 분석 ---
    print("\n" + "="*60)
    print("🔍 핵심 분석 결과")
    print("="*60)
    print("""
    - outro=0일 때: gamma_delta=0이므로 fused_gamma = DPCE_gamma
      → 출력은 DPCE와 동일해야 함
    
    - outro=xavier일 때: gamma_delta ≠ 0이므로 fused_gamma = DPCE_gamma + delta
      → 출력은 DPCE와 약간 다름
    
    - 만약 outro=0인데 출력이 원본과 같다면:
      → DPCE가 gamma ≈ 1을 출력한 것 (입력이 이미 밝거나 문제 있음)
    """)

if __name__ == "__main__":
    test_initial_state()
