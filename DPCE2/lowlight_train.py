import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
import dataloader
import model
import Myloss 
from torchvision import transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)

    if config.load_pretrain:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location=device))
        print("Pretrained model loaded.")

    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available() 
    )

    loss_fn = Myloss.TotalLoss(
        w_light=config.w_light,
        w_sfp=config.w_sfp,
        w_gamma=config.w_gamma
    ).to(device)

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()
    print("Starting training...")

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            
            img_lowlight = img_lowlight.to(device)

            optimizer.zero_grad()

            enhanced_image, gamma_map = DCE_net(img_lowlight)

            total_loss, loss_components = loss_fn(img_lowlight, enhanced_image, gamma_map)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], "
                      f"Iter [{iteration+1}/{len(train_loader)}], "
                      f"Total Loss: {loss_components['total_loss']:.4f} | "
                      f"Light: {loss_components['loss_light']:.4f} | "
                      f"SFP: {loss_components['loss_sfp']:.4f} | "
                      f"Gamma: {loss_components['loss_gamma']:.4f}")

            if ((iteration + 1) % config.snapshot_iter) == 0:
                snapshot_path = os.path.join(config.snapshots_folder, f"Epoch_{epoch+1}_Iter_{iteration+1}.pth")
                torch.save(DCE_net.state_dict(), snapshot_path)
    
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 경로 및 디렉토리
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data_zerodce_SIH/", help="Path to low-light training images")
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/", help="Folder to save snapshots")
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/your_model.pth", help="Path to a pretrained model")

    # 학습 하이퍼파라미터
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay for Adam optimizer")
    parser.add_argument('--grad_clip_norm', type=float, default=0.1, help="Gradient clipping norm")
    parser.add_argument('--num_epochs', type=int, default=200, help="Total number of training epochs")
    parser.add_argument('--train_batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument('--load_pretrain', action='store_true', help="Flag to load a pretrained model")

    # 손실 함수 가중치 (새로 추가)
    parser.add_argument('--w_light', type=float, default=1.0, help="Weight for L_light loss")
    parser.add_argument('--w_sfp', type=float, default=2.0, help="Weight for L_sfp (Self-Feature) loss")
    parser.add_argument('--w_gamma', type=float, default=0.01, help="Weight for L_gamma loss")

    # 로깅 및 저장 주기
    parser.add_argument('--display_iter', type=int, default=100, help="Iterations to display training loss")
    parser.add_argument('--snapshot_iter', type=int, default=200, help="Iterations to save a model snapshot")
    
    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
