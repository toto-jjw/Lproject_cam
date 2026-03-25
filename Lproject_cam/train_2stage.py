# Lproject_cam/train_2stage.py
# 2-Stage Training Script for DimCam
# Stage 1: freeze_backbone=True (Transformer만 학습)
# Stage 2: freeze_backbone=False (전체 모델 fine-tuning)

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
    
    def reset(self):
        """Stage 전환 시 카운터 리셋"""
        self.counter = 0
        self.early_stop = False


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs, train_loader_len):
    warmup_steps = warmup_epochs * train_loader_len
    total_steps = total_epochs * train_loader_len
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def freeze_backbone(model, freeze=True):
    """DPCE-Net backbone freeze/unfreeze"""
    for param in model.dce_net.parameters():
        param.requires_grad = not freeze
    
    status = "❄️ FROZEN" if freeze else "🔥 UNFROZEN"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  DPCE-Net backbone: {status}")
    print(f"  Trainable parameters: {trainable:,}")


def train_one_epoch(epoch, stage, dimcam_model, train_loader, criterion, 
                    optimizer, lr_scheduler, scaler, opt, device, writer, global_epoch):
    """단일 에폭 학습"""
    dimcam_model.train()
    
    # Stage 1에서는 backbone freeze 유지
    if stage == 1:
        dimcam_model.dce_net.eval()
    
    total_train_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Stage {stage} | Epoch {epoch+1} [Train]")
    
    optimizer.zero_grad(set_to_none=True)
    
    for i, batch_data in enumerate(train_pbar):
        img_l, img_r, ref_l, ref_r, depth_l, depth_r, calib_data, exposures = batch_data
        img_l, img_r = img_l.to(device), img_r.to(device)
        ref_l, ref_r = ref_l.to(device), ref_r.to(device)
        if depth_l is not None:
            depth_l = depth_l.to(device)
        if depth_r is not None:
            depth_r = depth_r.to(device)
        
        with autocast(enabled=(device.type == 'cuda')):
            outputs = dimcam_model(img_l, img_r, ref_l, ref_r, depth_l, calib_data)
            
            if len(outputs) == 9:
                pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r, _, _ = outputs
            else:
                pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r = outputs
            
            loss, loss_dict = criterion(
                img_l, img_r, pred_l, pred_r, 
                fused_gamma_l, fused_gamma_r, depth_l, depth_r, calib_data,
                exposure=exposures
            )
            loss = loss / opt.grad_accum_steps
        
        scaler.scale(loss).backward()
        
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
            
            if (i + 1) % (opt.grad_accum_steps * 10) == 0:
                torch.cuda.empty_cache()

        total_train_loss += loss.item() * opt.grad_accum_steps
        
        current_lr = lr_scheduler.get_last_lr()[0]
        train_pbar.set_postfix(
            loss=f"{loss.item():.4f}", 
            lr=f"{current_lr:.2e}",
            stage=stage
        )
        
        global_step = global_epoch * len(train_loader) + i
        for key, value in loss_dict.items():
            if value > 0:
                writer.add_scalar(f'Loss/{key}', value, global_step)
        writer.add_scalar('LearningRate/step', current_lr, global_step)
        writer.add_scalar('Stage', stage, global_step)

    return total_train_loss / len(train_loader)


def validate(epoch, stage, dimcam_model, val_loader, criterion, device, writer, global_epoch):
    """검증"""
    dimcam_model.eval()
    total_val_loss = 0
    val_loss_components = {}
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Stage {stage} | Epoch {epoch+1} [Val]")
        for batch_data in val_pbar:
            img_l, img_r, ref_l, ref_r, depth_l, depth_r, calib_data, exposures = batch_data
            img_l, img_r = img_l.to(device), img_r.to(device)
            ref_l, ref_r = ref_l.to(device), ref_r.to(device)
            if depth_l is not None:
                depth_l = depth_l.to(device)
            if depth_r is not None:
                depth_r = depth_r.to(device)
            
            with autocast(enabled=False):
                outputs = dimcam_model(img_l, img_r, ref_l, ref_r, depth_l, calib_data)
                
                if len(outputs) == 9:
                    pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r, _, _ = outputs
                else:
                    pred_l, pred_r, depth, _, _, fused_gamma_l, fused_gamma_r = outputs
                
                loss, loss_dict = criterion(
                    img_l, img_r, pred_l, pred_r, 
                    fused_gamma_l, fused_gamma_r, depth_l, depth_r, calib_data,
                    exposure=exposures
                )
            total_val_loss += loss.item()
            
            for key, value in loss_dict.items():
                val_loss_components[key] = val_loss_components.get(key, 0) + value
                
            val_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_val_loss = total_val_loss / len(val_loader)
    writer.add_scalar('Loss/validation_epoch', avg_val_loss, global_epoch)
    
    return avg_val_loss, val_loss_components


def train_2stage(opt):
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
    vis_loader = DataLoader(
        vis_subset, batch_size=1, shuffle=False,
        collate_fn=dataloader.LunarStereoDataset.collate_fn
    )
    
    print(f"  Train: {len(train_dataset)} samples ({train_dataset.get_scene_count()} scenes)")
    print(f"  Val: {len(val_dataset)} samples ({val_dataset.get_scene_count()} scenes)")

    # --- 모델 초기화 ---
    print("\nInitializing DimCamEnhancer...")
    
    dimcam_model = model.DimCamEnhancer(
        img_size=opt.img_height,
        embed_dim=opt.embed_dim,
        num_blocks=opt.num_blocks,
        use_grayscale=opt.use_grayscale,
        residual_scale=opt.residual_scale,
        gamma_min=opt.gamma_min,
        gamma_max=opt.gamma_max,
        use_noise_mask=opt.use_noise_mask,
        noise_hidden_channels=opt.noise_hidden_channels,
    ).to(device)

    # DPCE 가중치 로드
    if dpce_weights_path and os.path.exists(dpce_weights_path):
        print(f"Loading pretrained DPCE-Net weights from: {dpce_weights_path}")
        dimcam_model.dce_net.load_state_dict(torch.load(dpce_weights_path, map_location=device))
    else:
        print("Warning: Pretrained DPCE-Net weights not found.")
    
    # --- 손실 함수 초기화 ---
    light_loss_params = {
        'patch_size': opt.light_patch_size,
        'num_patches': opt.light_num_patches,
        'target_L': opt.light_target_L,
        'lambda_L': opt.light_lambda_L
    }

    criterion = Myloss.DimCamLoss(
        device=device,
        lambda_stereo=opt.lambda_stereo,
        lambda_depth=opt.lambda_depth,
        w_light=opt.w_light,
        w_sfp=opt.w_sfp,
        w_gamma=opt.w_gamma,
        w_color=opt.w_color,
        **light_loss_params
    ).to(device)

    # --- Early Stopper ---
    early_stopper = EarlyStopping(
        patience=opt.early_stop_patience,
        min_delta=opt.early_stop_delta
    )

    global_epoch = 0  # 전체 epoch 카운터 (시각화 및 로깅용)

    # =============================================
    # ★★★ STAGE 1: Freeze Backbone (Transformer만 학습) ★★★
    # =============================================
    print("\n" + "="*70)
    print("★★★ STAGE 1: Training Transformer Only (DPCE-Net Frozen) ★★★")
    print("="*70)
    
    freeze_backbone(dimcam_model, freeze=True)
    
    # Stage 1 옵티마이저 (높은 LR)
    optimizer_s1 = optim.AdamW(
        filter(lambda p: p.requires_grad, dimcam_model.parameters()), 
        lr=opt.lr_stage1, 
        weight_decay=opt.weight_decay
    )
    
    lr_scheduler_s1 = get_warmup_scheduler(
        optimizer_s1, 
        warmup_epochs=opt.warmup_epochs,
        total_epochs=opt.epochs_stage1,
        train_loader_len=len(train_loader)
    )
    
    print(f"Stage 1 Settings:")
    print(f"  - Epochs: {opt.epochs_stage1}")
    print(f"  - Learning Rate: {opt.lr_stage1}")
    print(f"  - Warmup Epochs: {opt.warmup_epochs}")
    
    best_val_loss_s1 = float('inf')
    
    for epoch in range(opt.epochs_stage1):
        # 학습
        avg_train_loss = train_one_epoch(
            epoch, 1, dimcam_model, train_loader, criterion,
            optimizer_s1, lr_scheduler_s1, scaler, opt, device, writer, global_epoch
        )
        
        # 검증
        avg_val_loss, val_loss_components = validate(
            epoch, 1, dimcam_model, val_loader, criterion, device, writer, global_epoch
        )
        
        print(f"\n--- Stage 1 | Epoch [{epoch+1}/{opt.epochs_stage1}] ---")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 체크포인트 저장
        if avg_val_loss < best_val_loss_s1:
            best_val_loss_s1 = avg_val_loss
            torch.save(
                dimcam_model.state_dict(), 
                os.path.join(opt.snapshot_folder, "dimcam_stage1_best.pth")
            )
            print(f"  ✅ Saved Stage 1 best model (val_loss: {best_val_loss_s1:.4f})")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                dimcam_model.state_dict(), 
                os.path.join(opt.snapshot_folder, "dimcam_enhancer_best.pth")
            )
        
        # 시각화
        if (epoch + 1) % opt.vis_interval == 0:
            save_visualizations(dimcam_model, vis_loader, vis_indices, 
                               opt.snapshot_folder, global_epoch, device, stage=1)
        
        # Early Stopping 체크
        if early_stopper(avg_val_loss, epoch + 1):
            print(f"\nStage 1 stopped early at epoch {epoch+1}")
            break
        
        global_epoch += 1
    
    # Stage 1 완료 후 best 모델 로드
    print(f"\n✅ Stage 1 Complete! Best val loss: {best_val_loss_s1:.4f}")
    dimcam_model.load_state_dict(
        torch.load(os.path.join(opt.snapshot_folder, "dimcam_stage1_best.pth"))
    )

    # =============================================
    # ★★★ STAGE 2: Fine-tune All (전체 모델 학습) ★★★
    # =============================================
    print("\n" + "="*70)
    print("★★★ STAGE 2: Fine-tuning Entire Model (DPCE-Net Unfrozen) ★★★")
    print("="*70)
    
    freeze_backbone(dimcam_model, freeze=False)
    
    # Stage 2 옵티마이저 (낮은 LR)
    optimizer_s2 = optim.AdamW(
        filter(lambda p: p.requires_grad, dimcam_model.parameters()), 
        lr=opt.lr_stage2,  # ★ 더 낮은 LR
        weight_decay=opt.weight_decay
    )
    
    lr_scheduler_s2 = get_warmup_scheduler(
        optimizer_s2, 
        warmup_epochs=opt.warmup_epochs_stage2,
        total_epochs=opt.epochs_stage2,
        train_loader_len=len(train_loader)
    )
    
    print(f"Stage 2 Settings:")
    print(f"  - Epochs: {opt.epochs_stage2}")
    print(f"  - Learning Rate: {opt.lr_stage2} (10x lower than Stage 1)")
    print(f"  - Warmup Epochs: {opt.warmup_epochs_stage2}")
    
    # Early stopper 리셋
    early_stopper.reset()
    early_stopper.best_score = best_val_loss  # 전체 best 유지
    
    best_val_loss_s2 = float('inf')
    
    for epoch in range(opt.epochs_stage2):
        # 학습
        avg_train_loss = train_one_epoch(
            epoch, 2, dimcam_model, train_loader, criterion,
            optimizer_s2, lr_scheduler_s2, scaler, opt, device, writer, global_epoch
        )
        
        # 검증
        avg_val_loss, val_loss_components = validate(
            epoch, 2, dimcam_model, val_loader, criterion, device, writer, global_epoch
        )
        
        print(f"\n--- Stage 2 | Epoch [{epoch+1}/{opt.epochs_stage2}] ---")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 체크포인트 저장
        if avg_val_loss < best_val_loss_s2:
            best_val_loss_s2 = avg_val_loss
            torch.save(
                dimcam_model.state_dict(), 
                os.path.join(opt.snapshot_folder, "dimcam_stage2_best.pth")
            )
            print(f"  ✅ Saved Stage 2 best model (val_loss: {best_val_loss_s2:.4f})")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                dimcam_model.state_dict(), 
                os.path.join(opt.snapshot_folder, "dimcam_enhancer_best.pth")
            )
            print(f"  🏆 New overall best model!")
        
        # 주기적 저장
        if (epoch + 1) % opt.save_interval == 0:
            torch.save(
                dimcam_model.state_dict(), 
                os.path.join(opt.snapshot_folder, f"dimcam_stage2_epoch_{epoch+1}.pth")
            )
        
        # 시각화
        if (epoch + 1) % opt.vis_interval == 0:
            save_visualizations(dimcam_model, vis_loader, vis_indices, 
                               opt.snapshot_folder, global_epoch, device, stage=2)
        
        # Early Stopping 체크
        if early_stopper(avg_val_loss, epoch + 1):
            print(f"\nStage 2 stopped early at epoch {epoch+1}")
            break
        
        global_epoch += 1
    
    # =============================================
    # ★★★ 최종 결과 ★★★
    # =============================================
    writer.close()
    
    print("\n" + "="*70)
    print("★★★ 2-STAGE TRAINING COMPLETE ★★★")
    print("="*70)
    print(f"Stage 1 Best Val Loss: {best_val_loss_s1:.4f}")
    print(f"Stage 2 Best Val Loss: {best_val_loss_s2:.4f}")
    print(f"Overall Best Val Loss: {best_val_loss:.4f}")
    print(f"\nSaved Models:")
    print(f"  - dimcam_stage1_best.pth (Stage 1 best)")
    print(f"  - dimcam_stage2_best.pth (Stage 2 best)")
    print(f"  - dimcam_enhancer_best.pth (Overall best)")
    print("="*70)


def save_visualizations(dimcam_model, vis_loader, vis_indices, snapshot_folder, 
                        global_epoch, device, stage):
    """시각화 저장"""
    vis_folder = os.path.join(snapshot_folder, 'visualizations')
    os.makedirs(vis_folder, exist_ok=True)
    
    dimcam_model.eval()
    with torch.no_grad():
        for vis_i, vis_data in enumerate(vis_loader):
            img_l, img_r, ref_l, ref_r, depth_l, depth_r, calib_data_vis, exposures = vis_data
            img_l, img_r = img_l.to(device), img_r.to(device)
            ref_l, ref_r = ref_l.to(device), ref_r.to(device)
            if depth_l is not None:
                depth_l = depth_l.to(device)
            
            outputs = dimcam_model(img_l, img_r, ref_l, ref_r, depth_l, calib_data_vis)
            
            if len(outputs) == 9:
                pred_l, _, depth, dpce_only_l, _, gamma_l, _, _, _ = outputs
            else:
                pred_l, _, depth, dpce_only_l, _, gamma_l, _ = outputs
            
            exp_str = exposures[0] if isinstance(exposures, list) else exposures
            prefix = f'stage{stage}_epoch_{global_epoch+1:03d}_img_{vis_indices[vis_i]:04d}_{exp_str}'
            
            # 비교 그리드
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
            
            # Gamma map
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-Stage Training for DimCam")
    
    # --- 경로 설정 ---
    parser.add_argument('--data_path', type=str, default='~/Downloads/lunardataset2/')
    parser.add_argument('--snapshot_folder', type=str, default='snapshots_dimcam_2stage/')
    parser.add_argument('--dpce_weights_path', type=str, default='../DPCE2/snapshots/Epoch_200_original.pth')

    # --- 데이터 설정 ---
    parser.add_argument('--use_all_exposures', type=str2bool, default=True)
    parser.add_argument('--reference_exposure', type=str, default='200ms')
    parser.add_argument('--use_precomputed_depth', type=str2bool, default=True)

    # --- 모델 하이퍼파라미터 ---
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--use_grayscale', type=str2bool, default=False)
    parser.add_argument('--residual_scale', type=float, default=1.0)
    parser.add_argument('--gamma_min', type=float, default=None)
    parser.add_argument('--gamma_max', type=float, default=None)
    parser.add_argument('--use_noise_mask', type=str2bool, default=False)
    parser.add_argument('--noise_hidden_channels', type=int, default=32)

    # --- ★★★ 2-Stage 학습 하이퍼파라미터 ★★★ ---
    # Stage 1: Transformer만 학습 (높은 LR)
    parser.add_argument('--epochs_stage1', type=int, default=50,
                        help="Number of epochs for Stage 1 (Transformer only)")
    parser.add_argument('--lr_stage1', type=float, default=1e-4,
                        help="Learning rate for Stage 1")
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help="Warmup epochs for Stage 1")
    
    # Stage 2: 전체 모델 fine-tuning (낮은 LR)
    parser.add_argument('--epochs_stage2', type=int, default=30,
                        help="Number of epochs for Stage 2 (Full model)")
    parser.add_argument('--lr_stage2', type=float, default=1e-5,
                        help="Learning rate for Stage 2 (should be lower than Stage 1)")
    parser.add_argument('--warmup_epochs_stage2', type=int, default=2,
                        help="Warmup epochs for Stage 2")

    # --- 공통 학습 설정 ---
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)

    # --- Early Stopping ---
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--early_stop_delta', type=float, default=0.0001)

    # --- 손실 함수 가중치 ---
    parser.add_argument('--lambda_stereo', type=float, default=1.0)
    parser.add_argument('--lambda_depth', type=float, default=0.5)
    parser.add_argument('--w_light', type=float, default=0.1)
    parser.add_argument('--w_sfp', type=float, default=0.3)
    parser.add_argument('--w_gamma', type=float, default=0.001)
    parser.add_argument('--w_color', type=float, default=100.0)

    # --- Light Loss 파라미터 ---
    parser.add_argument('--light_patch_size', type=int, default=32)
    parser.add_argument('--light_num_patches', type=int, default=10)
    parser.add_argument('--light_target_L', type=float, default=0.6)
    parser.add_argument('--light_lambda_L', type=float, default=4.5)

    # --- 로깅 및 저장 ---
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--vis_interval', type=int, default=5)
    
    opt = parser.parse_args()
    
    print("\n" + "="*70)
    print("★★★ DimCam 2-STAGE TRAINING ★★★")
    print("="*70)
    print(f"\n[Stage 1] Transformer Only (DPCE Frozen)")
    print(f"  - Epochs: {opt.epochs_stage1}")
    print(f"  - Learning Rate: {opt.lr_stage1}")
    print(f"  - Warmup: {opt.warmup_epochs} epochs")
    print(f"\n[Stage 2] Full Model Fine-tuning (DPCE Unfrozen)")
    print(f"  - Epochs: {opt.epochs_stage2}")
    print(f"  - Learning Rate: {opt.lr_stage2}")
    print(f"  - Warmup: {opt.warmup_epochs_stage2} epochs")
    print(f"\n[Common Settings]")
    print(f"  - Batch Size: {opt.batch_size}")
    print(f"  - Early Stop Patience: {opt.early_stop_patience}")
    print("="*70 + "\n")
    
    train_2stage(opt)
