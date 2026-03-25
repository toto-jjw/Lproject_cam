# Lproject_cam/train.py
# Training Script for DimCam with FoundationStereo

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# 현재 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model
import Myloss
import dataloader


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


class EarlyStopping:
    """Early Stopping 클래스"""
    def __init__(self, patience=15, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️ Early stopping triggered! Best epoch: {self.best_epoch}")
                
        return self.early_stop


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs, train_loader_len):
    """Warmup + Cosine Annealing 스케줄러"""
    warmup_steps = warmup_epochs * train_loader_len
    total_steps = total_epochs * train_loader_len
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(opt.snapshot_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.snapshot_folder, 'logs'))
    best_val_loss = float('inf')
    
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # --- 데이터 로딩 ---
    print("Loading datasets...")
    data_path = os.path.expanduser(opt.data_path)
    dpce_weights_path = os.path.expanduser(opt.dpce_weights_path) if opt.dpce_weights_path else None

    # ★★★ 노출도 그룹화 데이터셋 사용 ★★★
    # use_all_exposures=True: 같은 장면의 모든 노출도 이미지를 개별 샘플로 사용
    # reference_exposure='100ms': depth map 계산에 사용할 기준 노출
    # use_precomputed_depth=True: pre-computed depth 사용 (효율성)
    train_dataset = dataloader.LunarStereoDataset(
        data_path, mode='train', transform=True, 
        img_height=opt.img_height, img_width=opt.img_width,
        use_all_exposures=opt.use_all_exposures,
        reference_exposure=opt.reference_exposure,
        use_precomputed_depth=opt.use_precomputed_depth
    )
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.num_workers, pin_memory=True, 
        collate_fn=dataloader.LunarStereoDataset.collate_fn
    )
    
    val_dataset = dataloader.LunarStereoDataset(
        data_path, mode='val', transform=True, 
        img_height=opt.img_height, img_width=opt.img_width,
        use_all_exposures=opt.use_all_exposures,
        reference_exposure=opt.reference_exposure,
        use_precomputed_depth=opt.use_precomputed_depth
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.num_workers, pin_memory=True, 
        collate_fn=dataloader.LunarStereoDataset.collate_fn
    )
    
    vis_indices = list(range(min(5, len(val_dataset))))
    vis_subset = torch.utils.data.Subset(val_dataset, vis_indices)
    # vis_loader에도 collate_fn 적용
    vis_loader = DataLoader(
        vis_subset, batch_size=1, shuffle=False,
        collate_fn=dataloader.LunarStereoDataset.collate_fn
    )
    
    print("Datasets loaded.")
    print(f"  Train: {len(train_dataset)} samples ({train_dataset.get_scene_count()} scenes)")
    print(f"  Val: {len(val_dataset)} samples ({val_dataset.get_scene_count()} scenes)")

    # --- 모델 초기화 ---
    print("Initializing DimCamEnhancer (Noise-Aware)...")
    
    dimcam_model = model.DimCamEnhancer(
        img_size=opt.img_height,
        embed_dim=opt.embed_dim,
        num_blocks=opt.num_blocks,
        use_grayscale=opt.use_grayscale,
        residual_scale=opt.residual_scale,
        gamma_min=opt.gamma_min,
        gamma_max=opt.gamma_max,
        use_noise_mask=opt.use_noise_mask,  # ★★★ 노이즈 마스크 활성화 ★★★
        noise_hidden_channels=opt.noise_hidden_channels,
    ).to(device)

    # DPCE 가중치 로드
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

    print("\n" + "="*60)
    print(f"Model: DimCamEnhancer (Noise-Aware, Pre-computed Depth)")
    print(f"Mode: {'Grayscale' if opt.use_grayscale else 'RGB'}")
    print(f"Noise Mask: {'Enabled' if opt.use_noise_mask else 'Disabled'}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Gamma Range: No limit (free)")
    print(f"Initial Residual Scale: {opt.residual_scale}")
    print("="*60 + "\n")
    
    # --- 손실 함수 초기화 ---
    print("Initializing DimCamLoss...")
    
    light_loss_params = {
        'patch_size': opt.light_patch_size,
        'num_patches': opt.light_num_patches,
        'target_L': opt.light_target_L,  # ★★★ 고정 목표값 0.6 ★★★
        'lambda_L': opt.light_lambda_L
    }

    criterion = Myloss.DimCamLoss(
        device=device,
        lambda_stereo=opt.lambda_stereo,
        lambda_depth=opt.lambda_depth,
        w_light=opt.w_light,
        w_sfp=opt.w_sfp,
        w_gamma=opt.w_gamma,
        w_color=opt.w_color,  # ★ RGB 채널 간 일관성
        **light_loss_params
    ).to(device)

    # --- 옵티마이저 및 스케줄러 ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, dimcam_model.parameters()), 
        lr=opt.lr, 
        weight_decay=opt.weight_decay
    )
    
    lr_scheduler = get_warmup_scheduler(
        optimizer, 
        warmup_epochs=opt.warmup_epochs,
        total_epochs=opt.epochs,
        train_loader_len=len(train_loader)
    )
    
    early_stopper = EarlyStopping(
        patience=opt.early_stop_patience,
        min_delta=opt.early_stop_delta
    )
    
    print("Model, Loss, Optimizer, and Scheduler are ready.")
    print(f"Warmup epochs: {opt.warmup_epochs}")
    print(f"Early stopping patience: {opt.early_stop_patience}")

    # --- 학습 루프 ---
    print("\n--- Training Started ---")
    print(f"Using Gradient Accumulation: {opt.grad_accum_steps} steps (effective batch size = {opt.batch_size * opt.grad_accum_steps})")
    
    for epoch in range(opt.epochs):
        dimcam_model.train()
        if opt.freeze_backbone:
            dimcam_model.dce_net.eval()

        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Train]")
        
        optimizer.zero_grad(set_to_none=True)  # ★ 루프 시작 전에 초기화
        
        for i, batch_data in enumerate(train_pbar):
            # ★★★ 새로운 dataloader 형식: 8개 반환값 (depth_l, depth_r 분리) ★★★
            img_l, img_r, ref_l, ref_r, depth_l, depth_r, calib_data, exposures = batch_data
            img_l, img_r = img_l.to(device), img_r.to(device)
            ref_l, ref_r = ref_l.to(device), ref_r.to(device)
            if depth_l is not None:
                depth_l = depth_l.to(device)
            if depth_r is not None:
                depth_r = depth_r.to(device)
            
            with autocast(enabled=(device.type == 'cuda')):
                # ★★★ Noise-Aware Model: 9개 출력 (노이즈 마스크는 시각화용) ★★★
                # ★★★ calib_data 전달하여 baseline, focal_length 사용 ★★★
                outputs = dimcam_model(img_l, img_r, ref_l, ref_r, depth_l, calib_data)
                
                # 출력 개수에 따라 처리 (하위 호환성)
                if len(outputs) == 9:
                    pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r, noise_mask_l, noise_mask_r = outputs
                else:
                    pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r = outputs
                    noise_mask_l, noise_mask_r = None, None
                
                # ★★★ Reference-based SFP: 극저조도에서 200ms 참조 사용 ★★★
                loss, loss_dict = criterion(
                    img_l, img_r, pred_l, pred_r, 
                    fused_gamma_l, fused_gamma_r, depth_l, depth_r, calib_data,
                    exposure=exposures,
                    ref_l=ref_l, ref_r=ref_r  # ★ 200ms 참조 이미지 전달
                )
                # ★ Gradient Accumulation: loss를 accumulation steps로 나눔
                loss = loss / opt.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            # ★ Gradient Accumulation: accum_steps마다 optimizer step 실행
            if (i + 1) % opt.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, dimcam_model.parameters()), 
                    opt.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                
                # ★ 메모리 정리 (주기적)
                if (i + 1) % (opt.grad_accum_steps * 10) == 0:
                    torch.cuda.empty_cache()

            total_train_loss += loss.item() * opt.grad_accum_steps  # ★ 원래 loss 스케일로 복원
            
            current_lr = lr_scheduler.get_last_lr()[0]
            residual_scale = dimcam_model.get_residual_scale()
            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                lr=f"{current_lr:.2e}",
                res_scale=f"{residual_scale:.3f}"
            )
            
            global_step = epoch * len(train_loader) + i
            for key, value in loss_dict.items():
                if value > 0:
                    writer.add_scalar(f'Loss/{key}', value, global_step)
            writer.add_scalar('LearningRate/step', current_lr, global_step)
            writer.add_scalar('ResidualScale/step', residual_scale, global_step)

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        print(f"\n--- Epoch [{epoch+1}/{opt.epochs}] Train Loss: {avg_train_loss:.4f} ---")

        # --- 검증 단계 ---
        dimcam_model.eval()
        total_val_loss = 0
        val_loss_components = {}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{opt.epochs} [Val]")
            for i, batch_data in enumerate(val_pbar):
                # ★★★ 새로운 dataloader 형식: 8개 반환값 ★★★
                img_l, img_r, ref_l, ref_r, depth_l, depth_r, calib_data, exposures = batch_data
                img_l, img_r = img_l.to(device), img_r.to(device)
                ref_l, ref_r = ref_l.to(device), ref_r.to(device)
                if depth_l is not None:
                    depth_l = depth_l.to(device)
                if depth_r is not None:
                    depth_r = depth_r.to(device)
                
                with autocast(enabled=False):
                    # ★★★ calib_data 전달하여 baseline, focal_length 사용 ★★★
                    outputs = dimcam_model(img_l, img_r, ref_l, ref_r, depth_l, calib_data)
                    
                    # 출력 개수에 따라 처리 (하위 호환성)
                    if len(outputs) == 9:
                        pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r, noise_mask_l, noise_mask_r = outputs
                    else:
                        pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r = outputs
                        noise_mask_l, noise_mask_r = None, None
                    
                    # ★★★ Reference-based SFP: 극저조도에서 200ms 참조 사용 ★★★
                    loss, loss_dict = criterion(
                        img_l, img_r, pred_l, pred_r, 
                        fused_gamma_l, fused_gamma_r, depth_l, depth_r, calib_data,
                        exposure=exposures,
                        ref_l=ref_l, ref_r=ref_r  # ★ 200ms 참조 이미지 전달
                    )
                total_val_loss += loss.item()
                
                for key, value in loss_dict.items():
                    val_loss_components[key] = val_loss_components.get(key, 0) + value
                    
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)
        
        print(f"--- Epoch [{epoch+1}/{opt.epochs}] Val Loss: {avg_val_loss:.4f} ---")
        print("  Loss breakdown:")
        for key, value in val_loss_components.items():
            avg_value = value / len(val_loader)
            print(f"    {key}: {avg_value:.4f}")
            writer.add_scalar(f'ValLoss/{key}', avg_value, epoch)

        # --- 시각화 ---
        if (epoch + 1) % opt.vis_interval == 0:
            print("--- Generating visualization images... ---")
            vis_folder = os.path.join(opt.snapshot_folder, 'visualizations')
            os.makedirs(vis_folder, exist_ok=True)
            with torch.no_grad():
                for vis_i, vis_data in enumerate(vis_loader):
                    # ★★★ 새로운 dataloader 형식: 8개 반환값 ★★★
                    img_l, img_r, ref_l, ref_r, depth_l, depth_r, calib_data_vis, exposures = vis_data
                    img_l, img_r = img_l.to(device), img_r.to(device)
                    ref_l, ref_r = ref_l.to(device), ref_r.to(device)
                    if depth_l is not None:
                        depth_l = depth_l.to(device)
                    # ★★★ calib_data 전달하여 baseline, focal_length 사용 ★★★
                    outputs = dimcam_model(img_l, img_r, ref_l, ref_r, depth_l, calib_data_vis)
                    
                    # ★★★ Noise-Aware 출력 처리 ★★★
                    if len(outputs) == 9:
                        pred_l, _, depth, dpce_only_l, _, gamma_l, _, noise_mask_l, _ = outputs
                    else:
                        pred_l, _, depth, dpce_only_l, _, gamma_l, _ = outputs
                        noise_mask_l = None
                    
                    # 노출 시간을 파일명에 포함
                    exp_str = exposures[0] if isinstance(exposures, list) else exposures
                    prefix = f'epoch_{epoch+1:03d}_img_{vis_indices[vis_i]:04d}_{exp_str}'
                    
                    # 비교 그리드 (원본 | DPCE | 최종)
                    comparison_grid = vutils.make_grid(
                        [img_l[0].cpu(), 
                         dpce_only_l[0].cpu().clamp(0, 1), 
                         pred_l[0].cpu().clamp(0, 1)], 
                        nrow=3, padding=2
                    )
                    vutils.save_image(
                        comparison_grid, 
                        os.path.join(vis_folder, f'{prefix}_comparison.png')
                    )
                    
                    # Gamma map 시각화 (동적 범위 사용)
                    gamma_tensor = gamma_l[0, 0:1].cpu()
                    g_min, g_max = gamma_tensor.min(), gamma_tensor.max()
                    if g_max - g_min > 1e-6:
                        gamma_vis = (gamma_tensor - g_min) / (g_max - g_min)
                    else:
                        gamma_vis = gamma_tensor
                    vutils.save_image(
                        gamma_vis.clamp(0, 1),
                        os.path.join(vis_folder, f'{prefix}_gamma.png')
                    )
                    
                    # Depth map 시각화 (FoundationStereo)
                    if depth is not None:
                        depth_vis = depth[0].cpu()
                        vutils.save_image(
                            depth_vis.clamp(0, 1),
                            os.path.join(vis_folder, f'{prefix}_depth.png')
                        )
                    
                    # ★★★ Noise Mask 시각화 (새로 추가) ★★★
                    if noise_mask_l is not None:
                        noise_vis = noise_mask_l[0].cpu()
                        vutils.save_image(
                            noise_vis.clamp(0, 1),
                            os.path.join(vis_folder, f'{prefix}_noise_mask.png')
                        )
            print("--- Visualization images saved. ---")

        # --- 체크포인트 저장 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                dimcam_model.state_dict(), 
                os.path.join(opt.snapshot_folder, "dimcam_enhancer_best.pth")
            )
            print(f"✅ Saved best model with val loss: {best_val_loss:.4f}")
        
        if (epoch+1) % opt.save_interval == 0:
            torch.save(
                dimcam_model.state_dict(), 
                os.path.join(opt.snapshot_folder, f"dimcam_enhancer_epoch_{epoch+1}.pth")
            )

        # --- Early Stopping 체크 ---
        if early_stopper(avg_val_loss, epoch + 1):
            print(f"\nTraining stopped early at epoch {epoch+1}")
            break
            
    writer.close()
    print("\n--- Training Finished ---")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {early_stopper.best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DimCam with Pre-computed Depth.")
    
    # --- 경로 설정 ---
    parser.add_argument('--data_path', type=str, default='~/Downloads/lunardataset2/')
    parser.add_argument('--snapshot_folder', type=str, default='snapshots_dimcam/')
    parser.add_argument('--dpce_weights_path', type=str, default='../DPCE2/snapshots/Epoch_200_original.pth')

    # --- ★★★ 노출도 그룹화 및 Depth 설정 ★★★ ---
    parser.add_argument('--use_all_exposures', type=str2bool, default=True,
                        help="Use all exposure times as individual samples (True) or only reference exposure (False)")
    parser.add_argument('--reference_exposure', type=str, default='200ms',
                        help="Reference exposure time for depth map calculation (e.g., '200ms')")
    parser.add_argument('--use_precomputed_depth', type=str2bool, default=True,
                        help="Use pre-computed depth maps (run precompute_depth.py first)")

    # --- 모델 하이퍼파라미터 ---
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=8)  # 기존과 동일
    parser.add_argument('--freeze_backbone', type=str2bool, default=True)
    
    parser.add_argument('--use_grayscale', type=str2bool, default=False)  # ★ RGB 모드 + Color Loss로 색상 유지
    parser.add_argument('--residual_scale', type=float, default=1.0)  # ★ 핵심 수정: 스케일링 제거
    # ★ Gamma 범위 제한 없음 (시각화용으로만 사용)
    parser.add_argument('--gamma_min', type=float, default=None)  # 제한 없음
    parser.add_argument('--gamma_max', type=float, default=None)  # 제한 없음
    
    # --- ★★★ 노이즈 마스크 설정 (DimCam 논문) ★★★ ---
    parser.add_argument('--use_noise_mask', type=str2bool, default=False,
                        help="Enable noise mask extraction and noise-aware enhancement")
    parser.add_argument('--noise_hidden_channels', type=int, default=32,
                        help="Hidden channels for noise detector network")

    # --- 학습 하이퍼파라미터 ---
    parser.add_argument('--epochs', type=int, default=100)  # DimCam2와 동일
    parser.add_argument('--batch_size', type=int, default=1)  # DimCam2와 동일
    parser.add_argument('--grad_accum_steps', type=int, default=1)  # DimCam2와 동일 (accumulation 없음)
    parser.add_argument('--lr', type=float, default=1e-5)  # DimCam2와 동일
    parser.add_argument('--weight_decay', type=float, default=0.05)  # DimCam2와 동일
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=0)  # DimCam2와 동일 (warmup 없음)

    # --- Early Stopping ---
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--early_stop_delta', type=float, default=0.0001)

    # --- 손실 함수 가중치 (DimCam2와 동일) ---
    parser.add_argument('--lambda_stereo', type=float, default=1.0)   # DimCam2와 동일
    parser.add_argument('--lambda_depth', type=float, default=0.5)    # DimCam2와 동일
    parser.add_argument('--w_light', type=float, default=0.1)         # DimCam2와 동일
    parser.add_argument('--w_sfp', type=float, default=0.3)           # DimCam2와 동일
    parser.add_argument('--w_gamma', type=float, default=0.001)       # DimCam2와 동일
    parser.add_argument('--w_color', type=float, default=100.0)         # ★ RGB 채널 간 일관성 (색상 유지)

    # --- Light Loss 파라미터 (DPCE와 동일한 고정 목표값) ---
    parser.add_argument('--light_patch_size', type=int, default=32)
    parser.add_argument('--light_num_patches', type=int, default=10)
    parser.add_argument('--light_target_L', type=float, default=0.6)  # ★★★ 고정 목표값 0.6 ★★★
    parser.add_argument('--light_lambda_L', type=float, default=4.5)  # ★ 기존 DPCE lambda_L=4.5

    # --- 로깅 및 저장 간격 ---
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--vis_interval', type=int, default=5)
    
    opt = parser.parse_args()
    
    print("\n" + "="*60)
    print("DimCam Training (Noise-Aware) with Pre-computed Depth")
    print("="*60)
    print(f"\nKey settings:")
    print(f"  - Grayscale Mode: {opt.use_grayscale}")
    print(f"  - Batch Size: {opt.batch_size} (effective: {opt.batch_size * opt.grad_accum_steps} with grad accum)")
    print(f"  - Gradient Accumulation Steps: {opt.grad_accum_steps}")
    print(f"  - Early stopping (patience={opt.early_stop_patience})")
    print(f"  - LR warmup ({opt.warmup_epochs} epochs)")
    print(f"\n★★★ Noise Mask Settings (DimCam Paper) ★★★")
    print(f"  - Use Noise Mask: {opt.use_noise_mask}")
    print(f"  - Noise Hidden Channels: {opt.noise_hidden_channels}")
    print(f"\n★★★ Depth & Exposure Settings ★★★")
    print(f"  - Use All Exposures: {opt.use_all_exposures}")
    print(f"  - Reference Exposure for Depth: {opt.reference_exposure}")
    print(f"  - Use Pre-computed Depth: {opt.use_precomputed_depth}")
    if opt.use_precomputed_depth:
        print(f"  - ⚡ Depth maps loaded from .npy files (no runtime computation)")
    else:
        print(f"  - ⚠️ No depth available - run precompute_depth.py first!")
    print("\n" + "="*60)
    print(f"Full OPTIONS: {vars(opt)}")
    print("="*60 + "\n")
    
    train(opt)
