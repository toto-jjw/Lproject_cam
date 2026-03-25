# Lproject_cam/precompute_depth.py
# ★★★ Depth Map Pre-computation Script ★★★
# 학습 전에 모든 100ms 이미지에 대해 depth map을 미리 계산하여 저장
# 이후 학습 시에는 저장된 depth map을 로드하여 사용

import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import argparse
import re

# 현재 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FoundationStereo 임포트
try:
    foundation_stereo_path = os.path.join(os.path.dirname(__file__), '..', 'FoundationStereo')
    sys.path.append(foundation_stereo_path)
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    print("Successfully imported FoundationStereo.")
except ImportError as e:
    print(f"Error: Could not import FoundationStereo: {e}")
    sys.exit(1)


class DepthPrecomputer:
    """
    Depth Map Pre-computer using FoundationStereo
    
    Usage:
        python precompute_depth.py --data_path ~/Downloads/lunardataset2/ \
                                   --ckpt_path ../FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth \
                                   --reference_exposure 100ms
    """
    
    def __init__(self, ckpt_path, device='cuda', target_size=512, iters=8):
        self.device = device
        self.target_size = target_size  # Fixed resolution to avoid OOM
        self.iters = iters
        
        # FoundationStereo 로드
        print(f"Loading FoundationStereo from: {ckpt_path}")
        cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), 'cfg.yaml'))
        self.model = FoundationStereo(cfg)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.to(device)
        self.model.eval()
        print("FoundationStereo loaded successfully.")
    
    def compute_depth(self, img_l, img_r):
        """
        Compute normalized depth map from stereo pair
        
        Args:
            img_l: Left image tensor [1, 3, H, W], range [0, 1]
            img_r: Right image tensor [1, 3, H, W], range [0, 1]
        
        Returns:
            depth_normalized: Normalized depth map [H, W], range [0, 1]
        """
        with torch.no_grad():
            B, C, H, W = img_l.shape
            
            # ★★★ Fixed resolution for memory efficiency ★★★
            # Always resize to target_size to prevent OOM regardless of input resolution
            new_H = new_W = self.target_size
            img_l_scaled = torch.nn.functional.interpolate(
                img_l, size=(new_H, new_W), mode='bilinear', align_corners=False
            )
            img_r_scaled = torch.nn.functional.interpolate(
                img_r, size=(new_H, new_W), mode='bilinear', align_corners=False
            )
            
            # Free original tensors immediately
            del img_l, img_r
            torch.cuda.empty_cache()
            
            # Convert to 0-255 range (FoundationStereo expects this)
            img_l_255 = img_l_scaled * 255.0
            img_r_255 = img_r_scaled * 255.0
            del img_l_scaled, img_r_scaled
            
            # Pad for divisibility
            padder = InputPadder(img_l_255.shape, divis_by=32, force_square=False)
            img_l_padded, img_r_padded = padder.pad(img_l_255, img_r_255)
            
            # Run FoundationStereo
            with torch.cuda.amp.autocast(enabled=True):
                disp = self.model.forward(
                    img_l_padded, img_r_padded,
                    iters=self.iters,
                    test_mode=True
                )
            
            # Unpad
            disp = padder.unpad(disp.float())  # [B, H_scaled, W_scaled]
            
            # Upscale back to training resolution (512x512)
            # Note: We keep at 512x512 since training uses this resolution
            # disp is already at target_size, no need to upscale
            
            # Normalize per-image to [0, 1]
            # Disparity: larger = closer, normalize so that closer objects have higher values
            disp_np = disp[0].cpu().numpy()
            valid_mask = (disp_np > 0) & np.isfinite(disp_np)
            
            if valid_mask.sum() > 0:
                d_min = disp_np[valid_mask].min()
                d_max = disp_np[valid_mask].max()
                if d_max > d_min:
                    depth_normalized = (disp_np - d_min) / (d_max - d_min + 1e-8)
                else:
                    depth_normalized = disp_np / (d_max + 1e-8)
            else:
                depth_normalized = np.zeros_like(disp_np)
            
            depth_normalized = np.clip(depth_normalized, 0, 1).astype(np.float16)
        
        return depth_normalized
    
    def process_dataset(self, data_path, reference_exposure='100ms', output_suffix='_depth'):
        """
        Process entire dataset and save depth maps for BOTH left and right images
        
        Saves depth maps as .npy files next to the original images:
        - loc0_camL_200ms.png → loc0_camL_200ms_depth.npy (Left depth from L→R stereo)
        - loc1_camR_200ms.png → loc1_camR_200ms_depth.npy (Right depth from R→L stereo)
        """
        data_path = os.path.expanduser(data_path)
        
        # Find all reference exposure left images
        pattern = os.path.join(data_path, "View*", "View*", "Traverse*", "*m", f"*camL*{reference_exposure}*.png")
        left_images = sorted(glob.glob(pattern))
        
        print(f"Found {len(left_images)} reference ({reference_exposure}) stereo pairs to process.")
        print(f"Will generate depth maps for BOTH left and right images.")
        
        if len(left_images) == 0:
            print("No images found! Check the data path and reference exposure.")
            return
        
        # Process each stereo pair
        processed_l = 0
        processed_r = 0
        skipped = 0
        
        for left_path in tqdm(left_images, desc="Computing depth maps (L+R)"):
            # Construct right image path
            # ★★★ loc 번호가 +1인 right 이미지 찾기 ★★★
            # 예: loc310_camL_200ms.png → loc311_camR_200ms.png
            left_filename = os.path.basename(left_path)
            left_dir = os.path.dirname(left_path)
            
            loc_match = re.search(r'loc(\d+)_camL', left_filename)
            if loc_match:
                loc_num = int(loc_match.group(1))
                loc_num_right = loc_num + 1
                right_filename = left_filename.replace(f'loc{loc_num}_camL', f'loc{loc_num_right}_camR')
                right_path = os.path.join(left_dir, right_filename)
            else:
                # Fallback: 단순 치환
                right_path = left_path.replace('camL', 'camR')
            
            if not os.path.exists(right_path):
                print(f"Warning: Right image not found for {left_path}")
                skipped += 1
                continue
            
            # Depth output paths
            depth_path_l = left_path.replace('.png', f'{output_suffix}.npy')
            depth_path_r = right_path.replace('.png', f'{output_suffix}.npy')
            
            try:
                # Load images
                img_l = Image.open(left_path).convert('RGB')
                img_r = Image.open(right_path).convert('RGB')
                
                # Convert to tensor [1, 3, H, W]
                img_l_tensor = torch.from_numpy(np.array(img_l)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_r_tensor = torch.from_numpy(np.array(img_r)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
                img_l_tensor = img_l_tensor.to(self.device)
                img_r_tensor = img_r_tensor.to(self.device)
                
                # ★★★ Left Depth (L→R stereo matching) ★★★
                if not os.path.exists(depth_path_l):
                    depth_l = self.compute_depth(img_l_tensor, img_r_tensor)
                    np.save(depth_path_l, depth_l)
                    processed_l += 1
                
                # ★★★ Right Depth (R→L stereo matching, swap inputs) ★★★
                if not os.path.exists(depth_path_r):
                    depth_r = self.compute_depth(img_r_tensor, img_l_tensor)
                    np.save(depth_path_r, depth_r)
                    processed_r += 1
                
                # Clear GPU memory periodically
                if (processed_l + processed_r) % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {left_path}: {e}")
                skipped += 1
                continue
        
        print(f"\n✅ Depth computation complete!")
        print(f"   Left depth maps: {processed_l}")
        print(f"   Right depth maps: {processed_r}")
        print(f"   Skipped (already exists or error): {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute depth maps for training")
    parser.add_argument('--data_path', type=str, default='~/Downloads/lunardataset2/',
                        help="Path to dataset")
    parser.add_argument('--ckpt_path', type=str, 
                        default='../FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth',
                        help="Path to FoundationStereo checkpoint")
    parser.add_argument('--reference_exposure', type=str, default='100ms',
                        help="Reference exposure time for depth computation")
    parser.add_argument('--target_size', type=int, default=512,
                        help="Fixed resolution for depth computation (default: 512)")
    parser.add_argument('--iters', type=int, default=4,
                        help="Number of iterations for FoundationStereo")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Depth Map Pre-computation")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Reference exposure: {args.reference_exposure}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Iterations: {args.iters}")
    print("="*60 + "\n")
    
    # Initialize and run
    precomputer = DepthPrecomputer(
        ckpt_path=os.path.expanduser(args.ckpt_path),
        device=args.device,
        target_size=args.target_size,
        iters=args.iters
    )
    
    precomputer.process_dataset(
        data_path=args.data_path,
        reference_exposure=args.reference_exposure
    )


if __name__ == "__main__":
    main()
