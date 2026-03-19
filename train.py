import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from util.data_loader import create_dataloaders 
# Import 2 mạng riêng biệt từ model.py mới của bạn
# from models.multiheadv3 import UNetGenerator, PatchGANDiscriminator 
from models.pix2pixv2 import UNetGenerator, PatchGANDiscriminator
from util.loss import DiceLoss

# ==========================================
# 2. VÒNG LẶP HUẤN LUYỆN GAN
# ==========================================
def train_one_epoch(generator, discriminator, loader, opt_G, opt_D, criterions, device, lambda_pixel):
    generator.train()
    discriminator.train()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    
    # Biến tính Global Dice Score
    total_intersection = 0.0
    total_pred = 0.0
    total_target = 0.0

    pbar = tqdm(loader, desc="Training")
    
    # Lưu ý: target_label (nhãn phân loại cũ) giờ không dùng tới nữa, 
    # nhưng ta vẫn unpack ra để không bị lỗi DataLoader
    for context, target_img, target_label, _ in pbar:
        context = context.to(device)
        target_img = target_img.to(device).float()
        
        # Đảm bảo target_img có shape [B, 1, H, W]
        if target_img.dim() == 3:
            target_img = target_img.unsqueeze(1)

        # ----------------------------------
        # PHA 1: HUẤN LUYỆN DISCRIMINATOR (D)
        # Mục tiêu: Nhận biết rạch ròi Thật (1) và Giả (0)
        # ----------------------------------
        opt_D.zero_grad()
        
        # Sinh ra Mask giả (Dùng detach() để không lan truyền gradient về G lúc này)
        fake_mask_logits = generator(context)
        fake_mask = torch.sigmoid(fake_mask_logits) # Mask giả [0, 1]
        
        # D chấm điểm đồ thật
        pred_real = discriminator(context, target_img)
        # loss_D_real = criterions['gan'](pred_real, torch.ones_like(pred_real))
        loss_D_real = criterions['gan'](pred_real, torch.ones_like(pred_real) * 0.9)
        # D chấm điểm đồ giả
        pred_fake = discriminator(context, fake_mask.detach())
        loss_D_fake = criterions['gan'](pred_fake, torch.zeros_like(pred_fake))
        
        # Tổng Loss D (Trung bình của thật và giả)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # ----------------------------------
        # PHA 2: HUẤN LUYỆN GENERATOR (G)
        # Mục tiêu: Lừa D cho điểm 1 & Vẽ khớp từng pixel với target_img
        # ----------------------------------
        opt_G.zero_grad()
        
        # D chấm điểm lại đồ giả (Lần này KHÔNG detach để truyền gradient về G)
        pred_fake_for_G = discriminator(context, fake_mask)
        
        # G muốn lừa D rằng đây là đồ thật (Target = 1)
        # loss_G_GAN = criterions['gan'](pred_fake_for_G, torch.ones_like(pred_fake_for_G))
        loss_G_GAN = criterions['gan'](pred_fake_for_G, torch.ones_like(pred_fake_for_G) * 0.9)
        # G phải vẽ giống ảnh thật (Tính Dice Loss trên logits)
        loss_G_pixel = criterions['pixel'](fake_mask_logits, target_img)
        
        # Tổng Loss G = Loss đánh lừa + Trọng số * Loss vẽ giống
        loss_G = loss_G_GAN + lambda_pixel * loss_G_pixel
        loss_G.backward()
        opt_G.step()

        # ----------------------------------
        # TÍNH TOÁN CHỈ SỐ
        # ----------------------------------
        with torch.no_grad():
            pred_binary = (fake_mask > 0.5).float()
            target_binary = (target_img > 0.5).float()
            
            intersection = (pred_binary * target_binary).sum().item()
            pred_sum = pred_binary.sum().item()
            target_sum = target_binary.sum().item()
            
            total_intersection += intersection
            total_pred += pred_sum
            total_target += target_sum
            
            batch_dice = (2. * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)

        total_g_loss += loss_G.item()
        total_d_loss += loss_D.item()

        pbar.set_postfix(
            D_loss=f"{loss_D.item():.4f}", 
            G_loss=f"{loss_G.item():.4f}",
            px_loss=f"{loss_G_pixel.item():.4f}",
            b_dice=f"{batch_dice:.4f}"
        )

    epoch_dice = (2. * total_intersection + 1e-6) / (total_pred + total_target + 1e-6)
    return total_g_loss / len(loader), total_d_loss / len(loader), epoch_dice

# ==========================================
# 3. VÒNG LẶP VALIDATION (Chỉ đánh giá Generator)
# ==========================================
@torch.no_grad()
def validate(generator, loader, criterions, device, lambda_pixel):
    generator.eval()
    total_loss = 0.0
    
    total_intersection = 0.0
    total_pred = 0.0
    total_target = 0.0

    pbar = tqdm(loader, desc="Validating")
    
    for context, target_img, _, _ in pbar:
        context = context.to(device)
        target_img = target_img.to(device).float()
        if target_img.dim() == 3:
            target_img = target_img.unsqueeze(1)
        
        fake_mask_logits = generator(context)
        
        # Ở bước val, chúng ta chỉ quan tâm nó vẽ chuẩn pixel đến đâu
        loss_pixel = criterions['pixel'](fake_mask_logits, target_img)
        # Giả lập loss tổng của G để dễ theo dõi
        loss = lambda_pixel * loss_pixel 
        total_loss += loss.item()

        fake_mask = torch.sigmoid(fake_mask_logits)
        pred_binary = (fake_mask > 0.5).float()
        target_binary = (target_img > 0.5).float()
        
        intersection = (pred_binary * target_binary).sum().item()
        pred_sum = pred_binary.sum().item()
        target_sum = target_binary.sum().item()
        
        total_intersection += intersection
        total_pred += pred_sum
        total_target += target_sum
        
        batch_dice = (2. * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)

        pbar.set_postfix(
            val_loss=f"{loss.item():.4f}", 
            b_dice=f"{batch_dice:.4f}"
        )

    epoch_dice = (2. * total_intersection + 1e-6) / (total_pred + total_target + 1e-6)
    return total_loss / len(loader), epoch_dice

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

    # Khởi tạo 2 mạng
    generator = UNetGenerator(input_channels=1, output_channels=1).to(device)
    discriminator = PatchGANDiscriminator(input_channels=1, mask_channels=1).to(device)

    criterions = {
        'pixel': DiceLoss(), 
        'gan': nn.BCEWithLogitsLoss() # Loss dùng để đánh giá Thật/Giả cho PatchGAN
    }
    
    # Trọng số Pixel Loss. Thường trong Pix2Pix, Pixel Loss phải LỚN HƠN RẤT NHIỀU so với GAN loss.
    # Đề xuất: Đặt weight_pixel trong train.yaml khoảng 10.0 đến 100.0
    lambda_pixel = float(train_cfg.get("weight_pixel", 10.0))

    save_dir = train_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    last_checkpoint = os.path.join(save_dir, "last_gan.pt")
    best_checkpoint = os.path.join(save_dir, "best_gan.pt")

    # 2 Optimizer riêng biệt
    lr = float(train_cfg["lr"])
    opt_G = optim.AdamW(generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_D = optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)

    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    min_lr = float(train_cfg.get("min_lr", 1e-6)) 
    
    # 2 Scheduler riêng biệt
    warmup_G = LinearLR(opt_G, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_G = CosineAnnealingLR(opt_G, T_max=epochs - warmup_epochs, eta_min=min_lr)
    sch_G = SequentialLR(opt_G, schedulers=[warmup_G, cosine_G], milestones=[warmup_epochs])

    warmup_D = LinearLR(opt_D, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_D = CosineAnnealingLR(opt_D, T_max=epochs - warmup_epochs, eta_min=min_lr)
    sch_D = SequentialLR(opt_D, schedulers=[warmup_D, cosine_D], milestones=[warmup_epochs])

    start_epoch = 0
    best_dice = 0.0

    # Khôi phục checkpoint cho cả 2 mạng
    if os.path.exists(last_checkpoint):
        print(f"--- Đang khôi phục huấn luyện GAN từ: {last_checkpoint} ---")
        checkpoint = torch.load(last_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        discriminator.load_state_dict(checkpoint['disc_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        sch_G.load_state_dict(checkpoint['sch_G_state_dict'])
        sch_D.load_state_dict(checkpoint['sch_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"--- Tiếp tục từ epoch {start_epoch} ---")

    experiment_name = train_cfg.get('experiment_name', 'Pix2Pix_Pixel_Understanding')
    print(f"\n=== Bắt đầu Experiment: {experiment_name} ===") 
    print(f"Sử dụng lambda_pixel (Trọng số Dice so với GAN): {lambda_pixel}")

    for epoch in range(start_epoch, epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        
        train_g_loss, train_d_loss, train_dice = train_one_epoch(
            generator, discriminator, train_loader, opt_G, opt_D, criterions, device, lambda_pixel
        )
        val_loss, val_dice = validate(
            generator, val_loader, criterions, device, lambda_pixel
        )
        
        sch_G.step()
        sch_D.step()

        curr_lr = opt_G.param_groups[0]["lr"]
        print(f"D_Loss: {train_d_loss:.4f} | G_Loss: {train_g_loss:.4f} | Val G_Loss: {val_loss:.4f} | LR: {curr_lr:.6f}")
        print(f"Train Global Dice: {train_dice:.4f} | Val Global Dice: {val_dice:.4f}")
        
        checkpoint_data = {
            'epoch': epoch,
            'gen_state_dict': generator.state_dict(),
            'disc_state_dict': discriminator.state_dict(),
            'opt_G_state_dict': opt_G.state_dict(),
            'opt_D_state_dict': opt_D.state_dict(),
            'sch_G_state_dict': sch_G.state_dict(),
            'sch_D_state_dict': sch_D.state_dict(),
            'best_dice': best_dice,
        }
        torch.save(checkpoint_data, last_checkpoint)

        if val_dice > best_dice:
            best_dice = val_dice
            print(f"🔥 Best Model mới với Global Dice: {best_dice:.4f}! Lưu tại {best_checkpoint}")
            # Khi nộp bài (inference), ta chỉ cần Generator thôi nên chỉ lưu Gen weights
            torch.save(generator.state_dict(), best_checkpoint)

    final_path = os.path.join(save_dir, "final_gan_generator.pt")
    torch.save(generator.state_dict(), final_path)
    print(f"=== Hoàn tất huấn luyện {experiment_name}! ===")

if __name__ == "__main__":
    main()