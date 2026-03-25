# Lproject_cam/test.py
# Test Script for DimCam with FoundationStereo

import torch
import argparse
import torchvision.utils as vutils
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 현재 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model
import dataloader


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def test(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 경로 확장 ---
    data_path = os.path.expanduser(opt.data_path)
    weights_path = os.path.expanduser(opt.weights_path)

    # --- FoundationStereo checkpoint 경로 ---
    foundation_stereo_ckpt = None
    if opt.depth_mode == 'foundation_stereo':
        foundation_stereo_ckpt = os.path.expanduser(opt.foundation_stereo_ckpt)
        if not os.path.exists(foundation_stereo_ckpt):
            print(f"Warning: FoundationStereo checkpoint not found at {foundation_stereo_ckpt}")
            print("Falling back to MiDaS depth estimation.")
            opt.depth_mode = 'midas'

    # --- 모델 로드 ---
    print(f"Initializing DimCamEnhancer with depth_mode='{opt.depth_mode}'...")
    dimcam_model = model.DimCamEnhancer(
        patch_size=opt.patch_size,
        overlap=opt.overlap,
        embed_dim=opt.embed_dim,
        num_blocks=opt.num_blocks,
        lambda_depth=opt.lambda_depth,
        use_grayscale=opt.use_grayscale,
        residual_scale=opt.residual_scale,
        gamma_min=opt.gamma_min,
        gamma_max=opt.gamma_max,
        depth_mode=opt.depth_mode,
        foundation_stereo_ckpt=foundation_stereo_ckpt,
        foundation_stereo_iters=opt.foundation_stereo_iters,
        use_tiled_inference=opt.use_tiled_inference,
    ).to(device)
    
    print(f"Loading trained weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)

    # 불필요한 키 제거
    keys_to_remove = [k for k in state_dict if 'row_mask' in k]
    if keys_to_remove:
        print(f"Removing unexpected keys from state_dict: {keys_to_remove}")
        for key in keys_to_remove:
            del state_dict[key]

    dimcam_model.load_state_dict(state_dict, strict=False)
    dimcam_model.eval()
    print("Model loaded successfully.")

    # --- 테스트 데이터 로더 ---
    print("Loading test dataset...")
    test_dataset = dataloader.LunarStereoDataset(
        data_path, 
        mode=opt.test_mode,
        transform=True,
        img_height=opt.input_height, 
        img_width=opt.input_width
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=dataloader.LunarStereoDataset.collate_fn
    )
    print(f"Test dataset loaded: {len(test_dataset)} samples")

    # --- 출력 폴더 생성 ---
    output_folder = opt.output_folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'enhanced'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'comparison'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'gamma'), exist_ok=True)
    print(f"Saving results to {output_folder}")

    # --- 추론 및 저장 ---
    with torch.no_grad():
        for i, (img_l, img_r, calib_data) in enumerate(tqdm(test_loader, desc="Testing")):
            img_l, img_r = img_l.to(device), img_r.to(device)

            # Forward pass
            if opt.use_tiled_inference:
                # Tiled inference는 enhanced 이미지만 반환
                enhanced_l, enhanced_r = dimcam_model(img_l, img_r)
                depth = None
                dpce_only_l = None
                gamma_l = None
            else:
                outputs = dimcam_model(img_l, img_r)
                enhanced_l, enhanced_r, depth, dpce_only_l, _, gamma_l, _ = outputs
            
            base_name = f"{i:05d}"
            
            # --- Enhanced 이미지 저장 ---
            vutils.save_image(
                enhanced_l.clamp(0, 1), 
                os.path.join(output_folder, 'enhanced', f'{base_name}_left.png')
            )
            vutils.save_image(
                enhanced_r.clamp(0, 1), 
                os.path.join(output_folder, 'enhanced', f'{base_name}_right.png')
            )
            
            # --- 비교 이미지 저장 ---
            if dpce_only_l is not None:
                comparison_grid = vutils.make_grid(
                    [img_l[0].cpu(), 
                     dpce_only_l[0].cpu().clamp(0, 1), 
                     enhanced_l[0].cpu().clamp(0, 1)], 
                    nrow=3, padding=2
                )
            else:
                comparison_grid = vutils.make_grid(
                    [img_l[0].cpu(), 
                     enhanced_l[0].cpu().clamp(0, 1)], 
                    nrow=2, padding=2
                )
            vutils.save_image(
                comparison_grid, 
                os.path.join(output_folder, 'comparison', f'{base_name}_comparison.png')
            )
            
            # --- Depth map 저장 ---
            if depth is not None:
                # Normalize depth for visualization
                depth_vis = depth[0].cpu()
                vutils.save_image(
                    depth_vis.clamp(0, 1),
                    os.path.join(output_folder, 'depth', f'{base_name}_depth.png')
                )
                
                # Raw depth 저장 (optional)
                if opt.save_raw_depth:
                    np.save(
                        os.path.join(output_folder, 'depth', f'{base_name}_depth.npy'),
                        depth[0].cpu().numpy()
                    )
            
            # --- Gamma map 저장 ---
            if gamma_l is not None:
                gamma_vis = (gamma_l[0, 0:1].cpu() - opt.gamma_min) / (opt.gamma_max - opt.gamma_min)
                vutils.save_image(
                    gamma_vis.clamp(0, 1),
                    os.path.join(output_folder, 'gamma', f'{base_name}_gamma.png')
                )

    print(f"\n✅ Inference complete. Results saved to {output_folder}")
    print(f"   - Enhanced images: {output_folder}/enhanced/")
    print(f"   - Comparison images: {output_folder}/comparison/")
    print(f"   - Depth maps: {output_folder}/depth/")
    print(f"   - Gamma maps: {output_folder}/gamma/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DimCam with FoundationStereo.")
    
    # --- 경로 인자 ---
    parser.add_argument('--data_path', type=str, default='~/Downloads/lunardataset2',
                        help='Path to the dataset root')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained model weights (.pth file)')
    parser.add_argument('--output_folder', type=str, default='test_results_foundation/',
                        help='Folder to save the output images')
    parser.add_argument('--test_mode', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use for testing')
    
    # --- ★★★ FoundationStereo 설정 ★★★ ---
    parser.add_argument('--depth_mode', type=str, default='foundation_stereo',
                        choices=['foundation_stereo', 'midas'],
                        help="Depth estimation mode")
    parser.add_argument('--foundation_stereo_ckpt', type=str, 
                        default='../FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth',
                        help="Path to FoundationStereo checkpoint (11-33-40: ViT-Small, faster)")
    parser.add_argument('--foundation_stereo_iters', type=int, default=16,
                        help="Number of iterations for FoundationStereo inference")
    
    # --- Tiled Inference 파라미터 ---
    parser.add_argument('--use_tiled_inference', type=str2bool, default=False,
                        help='Use tiled inference for large images')
    parser.add_argument('--input_height', type=int, default=512,
                        help='Input image height')
    parser.add_argument('--input_width', type=int, default=512,
                        help='Input image width')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Patch size for tiled inference')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap size between patches')
    
    # --- 모델 하이퍼파라미터 (학습 시 설정과 일치) ---
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--lambda_depth', type=float, default=0.1)
    parser.add_argument('--use_grayscale', type=str2bool, default=True)
    parser.add_argument('--residual_scale', type=float, default=1.0)
    parser.add_argument('--gamma_min', type=float, default=0.3)
    parser.add_argument('--gamma_max', type=float, default=2.5)
    
    # --- 기타 인자 ---
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_raw_depth', type=str2bool, default=False,
                        help='Save raw depth as .npy file')

    opt = parser.parse_args()
    
    print("\n" + "="*60)
    print("DimCam Testing with FoundationStereo")
    print("="*60)
    print(f"Depth Mode: {opt.depth_mode}")
    print(f"Weights: {opt.weights_path}")
    print(f"Output: {opt.output_folder}")
    print("="*60 + "\n")
    
    test(opt)
