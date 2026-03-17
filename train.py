import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from util.data_loader import create_dataloaders 
# Đổi tên import sang model mới của chúng ta
from models.multiheadv2 import UNetMultiHeadV2
from util.loss import DiceLoss

# ==========================================
# 2. VÒNG LẶP HUẤN LUYỆN
# ==========================================
def train_one_epoch(model, loader, optimizer, criterions, device, loss_weights):
    model.train()
    total_loss = 0.0
    total_pixel_acc = 0.0  
    total_samples = 0

    pbar = tqdm(loader, desc="Training")
    
    for context, target_img, target_label, _ in pbar:
        context = context.to(device)
        target_img = target_img.to(device)
        target_label = target_label.to(device).long() 

        optimizer.zero_grad()
        
        # Forward pass: Lưu ý, lúc này pred_cls được sinh ra từ việc 
        # model tự động nối context và ảnh sinh ra (pred_pixel) ở bên trong!
        pred_pixel, pred_cls = model(context)
        
        # Tính Loss
        loss_pixel = criterions['pixel'](pred_pixel, target_img)
        loss_cls = criterions['cls'](pred_cls, target_label)
        
        # TỔNG LOSS: Đây chính là nơi mô hình bị "ép". 
        # Để loss_cls giảm, pred_pixel phải tạo ra hình thù giống chữ để lừa được Head 2.
        loss = loss_weights['pixel'] * loss_pixel + loss_weights['cls'] * loss_cls
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Tính Dice Score (Độ chính xác của riêng phần chữ)
        with torch.no_grad():
            pred_binary = (torch.sigmoid(pred_pixel) > 0.5).float()
            target_binary = (target_img > 0.5).float()
            
            intersection = (pred_binary * target_binary).sum().item()
            dice_score = (2. * intersection + 1e-6) / (pred_binary.sum().item() + target_binary.sum().item() + 1e-6)

        total_loss += loss.item()
        total_pixel_acc += dice_score  
        total_samples += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}", 
            pixel_dice=f"{dice_score:.4f}"
        )

    return total_loss / len(loader), total_pixel_acc / total_samples

# ==========================================
# 3. VÒNG LẶP VALIDATION
# ==========================================
@torch.no_grad()
def validate(model, loader, criterions, device, loss_weights):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Validating")
    
    for context, target_img, target_label, _ in pbar:
        context = context.to(device)
        target_img = target_img.to(device)
        target_label = target_label.to(device).long()
        
        pred_pixel, pred_cls = model(context)
        
        loss_pixel = criterions['pixel'](pred_pixel, target_img)
        loss_cls = criterions['cls'](pred_cls, target_label)
        loss = loss_weights['pixel'] * loss_pixel + loss_weights['cls'] * loss_cls
        total_loss += loss.item()

        pred_probs = torch.sigmoid(pred_pixel)
        pred_binary = (pred_probs > 0.5).float()
        target_binary = (target_img > 0.5).float()
        
        intersection = (pred_binary * target_binary).sum().item()
        denominator = pred_binary.sum().item() + target_binary.sum().item()
        dice = (2. * intersection + 1e-6) / (denominator + 1e-6)
        
        total_dice += dice
        total_samples += 1

        pbar.set_postfix(
            val_loss=f"{loss.item():.4f}", 
            val_pixel_dice=f"{dice:.4f}"
        )

    return total_loss / len(loader), total_dice / total_samples

# ==========================================
# 4. HÀM MAIN THỰC THI
# ==========================================
def main():
    with open("configs/data.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/train.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        train_csv=data_cfg["train_csv"],  
        val_csv=data_cfg["val_csv"],
        batch_size=data_cfg["batch_size"],
        fixed_height=data_cfg.get("fixed_height", 32),
        font_size=data_cfg.get("font_size", 24),
        font_path=data_cfg.get("font_path", None)
    )

    # Sử dụng Model Mới
    model = UNetMultiHeadV2().to(device) 

    criterions = {
        'pixel': DiceLoss(),          
        'cls': nn.CrossEntropyLoss()   
    }
    
    loss_weights = {
        'pixel': float(train_cfg.get("weight_pixel", 1.0)),
        'cls': float(train_cfg.get("weight_cls", 0.5)) # Trọng số ép buộc (có thể tăng lên 1.0 nếu model lười sinh chữ)
    }

    save_dir = train_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    last_checkpoint = os.path.join(save_dir, "last.pt")
    best_checkpoint = os.path.join(save_dir, "best.pt")

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(train_cfg["lr"]), 
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)) 
    )

    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    min_lr = float(train_cfg.get("min_lr", 1e-6)) 
    
    warmup_sch = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_epochs])

    start_epoch = 0
    best_dice = 0.0

    # Đã sửa lỗi logic khi khôi phục checkpoint (sử dụng best_dice thay vì best_loss)
    if os.path.exists(last_checkpoint):
        print(f"--- Đang khôi phục quá trình huấn luyện từ: {last_checkpoint} ---")
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"--- Tiếp tục từ epoch {start_epoch} ---")

    experiment_name = train_cfg.get('experiment_name', 'Kaggle_Pixel_Understanding')
    print(f"\n=== Bắt đầu Experiment: {experiment_name} ===") 

    for epoch in range(start_epoch, epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterions, device, loss_weights
        )
        val_loss, val_acc = validate(
            model, val_loader, criterions, device, loss_weights
        )
        
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {curr_lr:.6f}")
        print(f"Train Pixel Dice: {train_acc:.4f} | Val Pixel Dice: {val_acc:.4f}")
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice,
            'val_acc': val_acc,
        }
        torch.save(checkpoint_data, last_checkpoint)

        # Lưu model tốt nhất dựa trên Dice Score (càng cao càng tốt)
        if val_acc > best_dice:
            best_dice = val_acc
            print(f"Đã tìm thấy Best Model mới với Dice Score: {best_dice:.4f}! Lưu tại {best_checkpoint}")
            torch.save(model.state_dict(), best_checkpoint)

    final_path = os.path.join(save_dir, "final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"=== Hoàn tất huấn luyện {experiment_name}! ===")

if __name__ == "__main__":
    main()