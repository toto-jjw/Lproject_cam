# test.py (Final Version for NAFSSR-based DimCamEnhancer)

import torch
import argparse
import torchvision.utils as vutils
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# 최종 model, dataloader 임포트
import model
import dataloader as new_dataloader

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def test(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 경로 확장 ---
    data_path = os.path.expanduser(opt.data_path)
    weights_path = os.path.expanduser(opt.weights_path)

    # --- 2. ★★★ 모델 로드 (새로운 구조에 맞게 수정) ★★★ ---
    print("Initializing DimCamEnhancer model (NAFSSR-based architecture)...")
    # 학습 시 사용된 아키텍처 파라미터와 동일하게 모델을 초기화합니다.
    dimcam_model = model.DimCamEnhancer(
        # Wrapper 파라미터
        patch_size=opt.patch_size,
        overlap=opt.overlap,
        # Core 모델 파라미터
        embed_dim=opt.embed_dim,
        num_blocks=opt.num_blocks,
        lambda_depth=opt.lambda_depth
    ).to(device)
    
    print(f"Loading trained weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)

    # --- ★★★ 추가된 부분 시작 ★★★ ---
    # 저장된 state_dict에서 예상치 못한 키("row_mask")를 제거합니다.
    # 이 버퍼는 forward 시 동적으로 생성되므로, 로드할 필요가 없습니다.
    keys_to_remove = [k for k in state_dict if 'row_mask' in k]
    if keys_to_remove:
        print(f"Removing unexpected keys from state_dict: {keys_to_remove}")
        for key in keys_to_remove:
            del state_dict[key]
    # --- ★★★ 추가된 부분 끝 ★★★ --

    dimcam_model.load_state_dict(state_dict)
    dimcam_model.eval()
    print("Model loaded successfully.")

    # --- 3. ★★★ 테스트 데이터 로더 (고해상도 이미지 로드) ★★★ ---
    print("Loading test dataset (using 'validation' set)...")
    test_dataset = new_dataloader.LunarStereoDataset(
        data_path, 
        mode='val', # 검증셋을 테스트에 사용
        transform=True,
        img_height=opt.input_size, 
        img_width=opt.input_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=new_dataloader.LunarStereoDataset.collate_fn
    )
    print("Test dataset loaded.")

    # --- 4. ★★★ 출력 폴더 생성 및 추론/저장 로직 (단순화) ★★★ ---
    output_folder = opt.output_folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving results to {output_folder}")

    with torch.no_grad():
        for i, (img_l, img_r, _) in enumerate(tqdm(test_loader, desc="Testing images")):
            img_l, img_r = img_l.to(device), img_r.to(device)

            # 모델을 그냥 호출하면 Tiled Inference가 자동으로 실행됨
            enhanced_l, enhanced_r = dimcam_model(img_l, img_r)
            
            base_name = f"{i:05d}"
            
            # --- 결과 저장 ---
            vutils.save_image(img_l, os.path.join(output_folder, f'{base_name}_original_left.png'))
            vutils.save_image(enhanced_l.clamp(0, 1), os.path.join(output_folder, f'{base_name}_enhanced_left.png'))
            vutils.save_image(enhanced_r.clamp(0, 1), os.path.join(output_folder, f'{base_name}_enhanced_right.png'))

    print(f"\nInference complete. All results saved to {output_folder}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DimCamEnhancer with Tiled Inference.")
    
    # --- 경로 인자 ---
    parser.add_argument('--data_path', type=str, default='~/Downloads/lunardataset2', help='Path to the dataset root')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the trained model weights (.pth file)')
    parser.add_argument('--output_folder', type=str, default='test_results_final_tiled/', help='Folder to save the output images')
    
    # --- ★★★ Tiled Inference 파라미터 ★★★ ---
    parser.add_argument('--input_size', type=int, default=1024, help='Size of the large input image to test.')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size used during training. Must match the model.')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap size between patches for smooth merging.')
    
    # --- ★★★ Core 모델 하이퍼파라미터 (학습 시 설정과 반드시 일치) ★★★ ---
    parser.add_argument('--embed_dim', type=int, default=48, help="Embedding dimension used during training.")
    parser.add_argument('--num_blocks', type=int, default=1, help="Number of NAFBlocks used during training.")
    parser.add_argument('--lambda_depth', type=float, default=0.0, help="Must match the value used for training.")
    
    # --- 기타 인자 ---
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")

    opt = parser.parse_args()
    test(opt)
