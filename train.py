import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from util.data_loader import create_dataloaders 
from models.baseline import BaselineModel 

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")
    for context, target,_ in pbar:
        context, target = context.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(context)
    
        loss = criterion(outputs, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Tính Accuracy
        preds = (outputs > 0.5).float()
        correct += (preds == target).sum().item()
        total += target.num_elements() # hoặc target.size(0) tùy thuộc vào shape


        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader), correct / total if total > 0 else 0.0


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    pbar = tqdm(loader, desc="Validating")
    for context, target,_ in pbar:
        context, target = context.to(device), target.to(device)
        
        outputs = model(context)
        loss = criterion(outputs, target)
        
        total_loss += loss.item()

        # Tính Accuracy
        preds = (outputs > 0.5).float()
        correct += (preds == target).sum().item()
        total += target.num_elements()

        pbar.set_postfix(val_loss=loss.item())

    return total_loss / len(loader), correct / total if total > 0 else 0.0


def main():
    with open("configs/data.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/train.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        train_csv=data_cfg["train_csv"],  
        batch_size=data_cfg["batch_size"],
        fixed_height=data_cfg.get("fixed_height", 32), # Tham số từ config [cite: 13, 23]
        font_size=data_cfg.get("font_size", 24),
        font_path=data_cfg.get("font_path", None)
    )

    model = BaselineModel().to(device) 

    # Checkpoints setup
    save_dir = train_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    last_checkpoint = os.path.join(save_dir, "last.pt")
    best_checkpoint = os.path.join(save_dir, "best.pt")

    # optimizer
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

    # Resume logic
    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(last_checkpoint):
        print(f"--- Đang khôi phục quá trình huấn luyện từ: {last_checkpoint} ---")
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"--- Tiếp tục từ epoch {start_epoch} ---")

    # Loss function
    criterion = nn.BCELoss()

    # Train loop
    for epoch in range(start_epoch, epochs):
        print(f"\n--- Experiment: {train_cfg['experiment_name']} ---") # Dùng tên thí nghiệm
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {curr_lr:.6f}")
        print(f"                 Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        # Save Checkpoints
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'val_acc': val_acc,
        }
        torch.save(checkpoint_data, last_checkpoint)

        if val_loss < best_loss:
            best_loss = val_loss
            print(f"--- Đã tìm thấy Best Model mới với loss {best_loss:.6f} ---")
            torch.save(model.state_dict(), best_checkpoint)

    final_path = os.path.join(save_dir, "final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"--- Hoàn tất {train_cfg['experiment_name']}! ---")

if __name__ == "__main__":
    main()