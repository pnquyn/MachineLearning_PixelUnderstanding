import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split

class PixelUnderstandingDataset(Dataset):
    def __init__(self, data, split="train", fixed_height=32, font_size=24, font_path=None):
        """
        csv_path: Đường dẫn file csv 
        split: 'train' hoặc 'test' 
        """
        self.df = data
        self.split = split
        self.fixed_height = fixed_height
        self.font_size = font_size
        
        if font_path is None:
            paths = [
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", # Linux/Colab
                "C:\\Windows\\Fonts\\times.ttf", # Windows
                "/Library/Fonts/Times New Roman.ttf" # MacOS
            ]
            self.font_path = next((p for p in paths if os.path.exists(p)), None)
        else:
            self.font_path = font_path

        print(f"Loaded {len(self.df)} samples for {split} split.")

    def _text_to_image(self, text):
        """Chuyển đổi văn bản thành ảnh grayscale (L)"""
        if not isinstance(text, str): text = ""
        
        try:
            font = ImageFont.truetype(self.font_path, self.font_size) if self.font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Tính toán chiều rộng linh hoạt dựa trên text 
        mask = font.getmask(text)
        text_width, text_height = mask.size
        canvas_width = max(text_width + 10, 1) 
        
        # Tạo ảnh đen chữ trắng 
        img = Image.new('L', (canvas_width, self.fixed_height), color=0)
        draw = ImageDraw.Draw(img)
        
        # Căn giữa dọc
        y_offset = (self.fixed_height - text_height) // 2
        draw.text((5, y_offset), text, fill=255, font=font)
        
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['sample_id']
        context_img = self._text_to_image(row['context'])
        context_tensor = F.to_tensor(context_img)
        
        if 'target' in row:
            target_img = self._text_to_image(row['target'])
            target_tensor = F.to_tensor(target_img)
            return context_tensor, target_tensor, torch.tensor(sample_id, dtype=torch.long)

        return context_tensor, torch.tensor(sample_id, dtype=torch.long)

def collate_fn(batch):
    """
    Xử lý Batch cho ảnh có chiều rộng linh hoạt bằng cách Padding vùng đen (0) 
    """
    if len(batch[0]) == 3:
        contexts, targets, ids = zip(*batch)
        
        max_w_ctx = max(img.shape[2] for img in contexts)
        max_w_tgt = max(img.shape[2] for img in targets)
        
        padded_contexts = torch.stack([
            F.pad(img, (0, 0, max_w_ctx - img.shape[2], 0), fill=0) for img in contexts
        ])

        padded_targets = torch.stack([
            F.pad(img, (0, 0, max_w_tgt - img.shape[2], 0), fill=0) for img in targets
        ])
        
        return padded_contexts, padded_targets, torch.stack(ids)
    
    else: # Mode Test: (context, id)
        contexts, ids = zip(*batch)
        max_w = max(img.shape[2] for img in contexts)
        padded_contexts = torch.stack([
            F.pad(img, (0, 0, max_w - img.shape[2], 0), fill=0) for img in contexts
        ])
        return padded_contexts, torch.stack(ids)

def create_dataloaders(train_csv, batch_size=16, fixed_height=32, font_size=24, font_path=None, val_size=0.2):
    """
    Chia train_csv thành tập Train và Val với tỉ lệ cố định.
    """
    full_df = pd.read_csv(train_csv)

    train_df, val_df = train_test_split(
        full_df, 
        test_size=val_size, 
        random_state=42, 
        shuffle=True
    )
    
    print(f"--- Data Split: Train={len(train_df)} samples, Val={len(val_df)} samples ---")

    train_ds = PixelUnderstandingDataset(
        train_df, 
        split="train", 
        fixed_height=fixed_height, 
        font_size=font_size, 
        font_path=font_path
    )
    
    val_ds = PixelUnderstandingDataset(
        val_df, 
        split="train", 
        fixed_height=fixed_height, 
        font_size=font_size, 
        font_path=font_path
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader