# train.py (Final Version for NAFSSR-based DimCamEnhancer)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
from torch.cuda.amp import autocast, GradScaler

# 수정된 model.py와 통합 Myloss.py를 임포트
import model 
import Myloss 
import dataloader as new_dataloader

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(opt.snapshot_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.snapshot_folder, 'logs'))
    best_val_loss = float('inf')
    
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    print("Loading datasets...")
    data_path = os.path.expanduser(opt.data_path)
    dpce_weights_path = os.path.expanduser(opt.dpce_weights_path) if opt.dpce_weights_path else None

    train_dataset = new_dataloader.LunarStereoDataset(data_path, mode='train', transform=True, img_height=opt.img_height, img_width=opt.img_width)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, collate_fn=new_dataloader.LunarStereoDataset.collate_fn)
    
    val_dataset = new_dataloader.LunarStereoDataset(data_path, mode='val', transform=True, img_height=opt.img_height, img_width=opt.img_width)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True, collate_fn=new_dataloader.LunarStereoDataset.collate_fn)
    
    vis_indices = list(range(min(5, len(val_dataset))))
    vis_subset = torch.utils.data.Subset(val_dataset, vis_indices)
    vis_loader = DataLoader(vis_subset, batch_size=1, shuffle=False)
    print("Datasets loaded.")

    # --- 3. ★★★ 모델 초기화 수정 ★★★ ---
    print("Initializing DimCamEnhancer (NAFSSR-based architecture)...")
    dimcam_model = model.DimCamEnhancer(
        img_size=opt.img_height,
        embed_dim=opt.embed_dim,
        num_blocks=opt.num_blocks,
        lambda_depth=opt.lambda_depth,
        use_grayscale=opt.use_grayscale  # ★ Grayscale/RGB 모드 선택
    ).to(device)

    if dpce_weights_path and os.path.exists(dpce_weights_path):
        print(f"Loading pretrained DPCE-Net weights from: {dpce_weights_path}")
        dimcam_model.dce_net.load_state_dict(torch.load(dpce_weights_path, map_location=device))
    else:
        print("Warning: Pretrained DPCE-Net weights not found.")

    if opt.freeze_backbone:
        print("Freezing pretrained DPCE-Net backbone.")
        for param in dimcam_model.dce_net.parameters():
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in dimcam_model.parameters())
    trainable_params = sum(p.numel() for p in dimcam_model.parameters() if p.requires_grad)

    print("\n" + "="*50)
    print(f"Model: {dimcam_model.__class__.__name__} (NAFSSR-based)")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print("="*50 + "\n")
    
    # --- 4. ★★★ 손실 함수, 옵티마이저, 스케줄러 복구 ★★★ ---
    print("Initializing Rebalanced Integrated Loss, Optimizer, and Scheduler...")
    
    light_loss_params = {
        'patch_size': opt.light_patch_size,
        'num_patches': opt.light_num_patches,
        'target_L': opt.light_target_L,
        'lambda_L': opt.light_lambda_L
    }

    criterion = Myloss.DimCamLoss(
        device=device,
        lambda_stereo=opt.lambda_stereo, lambda_depth=opt.lambda_depth,
        w_light=opt.w_light, w_sfp=opt.w_sfp, w_gamma=opt.w_gamma,
        w_color=opt.w_color,  # ★ Color Loss 가중치
        **light_loss_params
    ).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, dimcam_model.parameters()), lr=opt.lr, weight_decay=0.05)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * opt.epochs, eta_min=1e-6)
    
    print("Model, Loss, Optimizer, and Scheduler are ready.")

    # --- 5. ★★★ 학습 및 검증 루프 (통합 손실 사용하도록 복구) ★★★ ---
    print("--- Training Started ---")
    for epoch in range(opt.epochs):
        dimcam_model.train()
        if opt.freeze_backbone: dimcam_model.dce_net.eval()
        if dimcam_model.depth_net is not None: dimcam_model.depth_net.eval()

        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Train]")
        
        for i, (img_l, img_r, calib_data) in enumerate(train_pbar):
            img_l, img_r = img_l.to(device), img_r.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(device.type == 'cuda')):
                outputs = dimcam_model(img_l, img_r)
                pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r = outputs
                loss, loss_dict = criterion(img_l, img_r, pred_l, pred_r, fused_gamma_l, fused_gamma_r, depth, calib_data)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, dimcam_model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            global_step = epoch * len(train_loader) + i
            for key, value in loss_dict.items():
                if value > 0: writer.add_scalar(f'Loss/{key}', value, global_step)
            writer.add_scalar('LearningRate/step', lr_scheduler.get_last_lr()[0], global_step)

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        print(f"--- Epoch [{epoch+1}/{opt.epochs}] Average Train Loss: {avg_train_loss:.4f} ---")

        # 검증 단계
        dimcam_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Val]")
            for i, (img_l, img_r, calib_data) in enumerate(val_pbar):
                img_l, img_r = img_l.to(device), img_r.to(device)
                with autocast(enabled=False):
                    outputs = dimcam_model(img_l, img_r)
                    pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r = outputs
                    loss, _ = criterion(img_l, img_r, pred_l, pred_r, fused_gamma_l, fused_gamma_r, depth, calib_data)
                total_val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)
        print(f"--- Epoch [{epoch+1}/{opt.epochs}] Average Validation Loss: {avg_val_loss:.4f} ---")

        # 시각화 단계
        if (epoch + 1) % opt.vis_interval == 0:
            print("--- Generating visualization images... ---")
            vis_folder = os.path.join(opt.snapshot_folder, 'visualizations')
            os.makedirs(vis_folder, exist_ok=True)
            with torch.no_grad():
                for vis_i, (img_l, img_r, _) in enumerate(vis_loader):
                    img_l, img_r = img_l.to(device), img_r.to(device)
                    outputs = dimcam_model(img_l, img_r)
                    pred_l, _, _, dpce_only_l, _, _, _ = outputs
                    prefix = f'epoch_{epoch+1:03d}_img_{vis_indices[vis_i]:04d}'
                    comparison_grid = vutils.make_grid([img_l[0].cpu(), dpce_only_l[0].cpu().clamp(0, 1), pred_l[0].cpu().clamp(0, 1)], nrow=3, padding=2)
                    vutils.save_image(comparison_grid, os.path.join(vis_folder, f'{prefix}_comparison.png'))
            print("--- Visualization images saved. ---")

        # 스냅샷 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(dimcam_model.state_dict(), os.path.join(opt.snapshot_folder, "dimcam_enhancer_best.pth"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        if (epoch+1) % opt.save_interval == 0:
            torch.save(dimcam_model.state_dict(), os.path.join(opt.snapshot_folder, f"dimcam_enhancer_epoch_{epoch+1}.pth"))
            
    writer.close()
    print("--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NAFSSR-based DimCamEnhancer.")
    
    # 경로 및 기본 설정
    parser.add_argument('--data_path', type=str, default='~/Downloads/lunardataset2/')
    parser.add_argument('--snapshot_folder', type=str, default='snapshots_dimcam_nafssr/')
    parser.add_argument('--dpce_weights_path', type=str, default='DPCE2/snapshots/Epoch_200_original.pth')

    # --- ★★★ DPCE 방식의 L_light를 위한 하이퍼파라미터 추가 ★★★ ---
    parser.add_argument('--light_patch_size', type=int, default=32)
    parser.add_argument('--light_num_patches', type=int, default=10)
    parser.add_argument('--light_target_L', type=float, default=0.6)
    parser.add_argument('--light_lambda_L', type=float, default=4.5)
    
    # 학습 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size. Reduce if OOM.") # 1
    parser.add_argument('--lr', type=float, default=1e-5) # 2D Conv 기반 모델은 조금 더 높은 LR을 사용해도 안정적일 수 있음 1e-5
    parser.add_argument('--num_workers', type=int, default=4)

    # --- ★★★ 모델 하이퍼파라미터 수정 ★★★ ---
    parser.add_argument('--img_height', type=int, default=512) # 128 512
    parser.add_argument('--img_width', type=int, default=512)
    # embed_dim은 이제 특징맵의 채널 수. NAFNet은 48, 64 등을 사용.
    parser.add_argument('--embed_dim', type=int, default=64, help="Embedding dimension (channel count).") #48 64
    parser.add_argument('--num_blocks', type=int, default=5, help="Number of self-refinement blocks.") # 1 6
    parser.add_argument('--freeze_backbone', type=str2bool, default=True, help="Freeze the DPCE-Net backbone.")
    parser.add_argument('--use_grayscale', type=str2bool, default=False, help="Use grayscale gamma (True) or RGB gamma (False).")

    # --- ★★★ 손실 함수 하이퍼파라미터 ★★★ ---
    parser.add_argument('--lambda_stereo', type=float, default=2.0, help="Weight for stereo consistency loss.") # 2 5 10
    parser.add_argument('--lambda_depth', type=float, default=0.1, help="Weight for depth consistency loss.") # 0 0.1 0.5
    parser.add_argument('--w_light', type=float, default=0.1, help='Weight for light consistency loss.')
    parser.add_argument('--w_sfp', type=float, default=0.2, help='Weight for perceptual (VGG) loss.')
    parser.add_argument('--w_gamma', type=float, default=0.001, help='Weight for gamma smoothness loss.')
    parser.add_argument('--w_color', type=float, default=100.0, help='Weight for color ratio preservation loss (RGB mode only).')

    # 로깅 및 저장 간격
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--vis_interval', type=int, default=5)
    
    opt = parser.parse_args()
    
    print("\n" + "="*50)
    print("Starting training with NAFSSR-based architecture and REBALANCED LOSS.")
    print(f"OPTIONS: {vars(opt)}")
    print("="*50 + "\n")
    
    train(opt)
