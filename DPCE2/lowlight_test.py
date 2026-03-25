import torch
import torch.nn as nn
import torchvision
import os
import argparse
import glob
import time
from PIL import Image
import numpy as np
import model

def test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load(config.weights_path, map_location=device))
    DCE_net.eval()
    print(f"Model loaded from {config.weights_path}")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    test_list = []
    for ext in image_extensions:
        test_list.extend(glob.glob(os.path.join(config.test_dir, ext)))
    
    if not test_list:
        print(f"No images found in {config.test_dir}")
        return

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        print(f"Created output directory: {config.output_dir}")

    total_time = 0
    with torch.no_grad(): 
        for image_path in test_list:
            print(f"Processing: {image_path}")

            # 이미지 로드 및 전처리
            try:
                data_lowlight = Image.open(image_path)
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue

            data_lowlight = (np.asarray(data_lowlight) / 255.0)
            data_lowlight = torch.from_numpy(data_lowlight).float()

            if data_lowlight.ndimension() == 2:  # 흑백 이미지를 3채널로 복사
                data_lowlight = data_lowlight.unsqueeze(2).repeat(1, 1, 3)
            
            # (H, W, C) -> (C, H, W) -> (1, C, H, W)
            data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).to(device)

            start_time = time.time()
            enhanced_image, _ = DCE_net(data_lowlight)
            end_time = time.time()
            
            inference_time = end_time - start_time
            total_time += inference_time
            print(f"  Inference time: {inference_time:.4f} seconds")

            image_name = os.path.basename(image_path)
            output_path = os.path.join(config.output_dir, image_name)
            torchvision.utils.save_image(enhanced_image, output_path)
            print(f"  Result saved to: {output_path}")

    print("\n--------------------")
    print(f"Testing finished. Processed {len(test_list)} images.")
    print(f"Average inference time: {total_time / len(test_list):.4f} seconds per image.")
    print("--------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dir', type=str, default='data/test_data/', help='Path to the directory containing test images')
    parser.add_argument('--weights_path', type=str, default='snapshots/Epoch_30_Iter_200.pth', help='Path to the pretrained model weights')
    parser.add_argument('--output_dir', type=str, default='data/result/', help='Directory to save enhanced images')

    config = parser.parse_args()

    test(config)

