import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split


def _pad_width_to_multiple_of_8(w):
    """Làm tròn width lên bội số của 8 (cần cho stride=2 x3 trong model)"""
    return ((w + 7) // 8) * 8


class PixelUnderstandingDataset(Dataset):
    def __init__(self, data, split="train", fixed_height=32, font_size=24, font_path=None):
        """
        data: DataFrame chứa dữ liệu
        split: 'train' hoặc 'test'
        """
        self.df = data.reset_index(drop=True)
        self.split = split
        self.fixed_height = fixed_height
        self.font_size = font_size
        
        # Tự động tìm font phù hợp theo OS
        if font_path is None or not os.path.exists(str(font_path)):
            paths = [
                "C:\\Windows\\Fonts\\times.ttf",       # Windows
                "C:\\Windows\\Fonts\\arial.ttf",        # Windows fallback
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",  # Linux/Colab
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
                "/Library/Fonts/Times New Roman.ttf",   # MacOS
            ]
            self.font_path = next((p for p in paths if os.path.exists(p)), None)
        else:
            self.font_path = font_path

        print(f"Loaded {len(self.df)} samples for {split} split. Font: {self.font_path}")

    def _text_to_image(self, text):
        """Chuyển đổi văn bản thành ảnh grayscale (L): nền đen, chữ trắng"""
        if not isinstance(text, str):
            text = ""
        
        try:
            font = ImageFont.truetype(self.font_path, self.font_size) if self.font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Tính toán chiều rộng linh hoạt dựa trên text
        mask = font.getmask(text)
        text_width, text_height = mask.size
        canvas_width = max(text_width + 10, 1) 
        
        # Tạo ảnh đen, chữ trắng
        img = Image.new('L', (canvas_width, self.fixed_height), color=0)
        draw = ImageDraw.Draw(img)
        
        # Căn giữa dọc
        y_offset = max((self.fixed_height - text_height) // 2, 0)
        draw.text((5, y_offset), text, fill=255, font=font)
        
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']  # FIX: dùng cột 'id' thay vì 'sample_id'
        context_img = self._text_to_image(row['context'])
        context_tensor = F.to_tensor(context_img)  # (1, H, W), giá trị [0, 1]
        
        if 'target' in row:
            target_img = self._text_to_image(row['target'])
            target_tensor = F.to_tensor(target_img)
            return context_tensor, target_tensor, torch.tensor(sample_id, dtype=torch.long)

        # Test mode: không có target
        return context_tensor, torch.tensor(sample_id, dtype=torch.long)


def collate_fn(batch):
    """
    Xử lý Batch: pad tất cả ảnh về cùng kích thước (bội 8) bằng vùng đen (0).
    Context và Target được pad về CÙNG width để model output khớp target.
    """
    if len(batch[0]) == 3:  # Train mode: (context, target, id)
        contexts, targets, ids = zip(*batch)
        
        # Pad cả context và target về CÙNG max width (bội 8)
        max_w = max(
            max(img.shape[2] for img in contexts),
            max(img.shape[2] for img in targets)
        )
        max_w = _pad_width_to_multiple_of_8(max_w)
        
        padded_contexts = torch.stack([
            F.pad(img, (0, 0, max_w - img.shape[2], 0), fill=0) for img in contexts
        ])
        padded_targets = torch.stack([
            F.pad(img, (0, 0, max_w - img.shape[2], 0), fill=0) for img in targets
        ])
        
        return padded_contexts, padded_targets, torch.stack(ids)
    
    else:  # Test mode: (context, id)
        contexts, ids = zip(*batch)
        max_w = max(img.shape[2] for img in contexts)
        max_w = _pad_width_to_multiple_of_8(max_w)
        
        padded_contexts = torch.stack([
            F.pad(img, (0, 0, max_w - img.shape[2], 0), fill=0) for img in contexts
        ])
        return padded_contexts, torch.stack(ids)


def create_dataloaders(train_csv, batch_size=16, fixed_height=32, font_size=24, font_path=None, val_size=0.2):
    """Chia train_csv thành tập Train và Val."""
    full_df = pd.read_csv(train_csv)

    train_df, val_df = train_test_split(
        full_df, 
        test_size=val_size, 
        random_state=42, 
        shuffle=True
    )
    
    print(f"--- Data Split: Train={len(train_df)} samples, Val={len(val_df)} samples ---")

    train_ds = PixelUnderstandingDataset(
        train_df, split="train",
        fixed_height=fixed_height, font_size=font_size, font_path=font_path
    )
    val_ds = PixelUnderstandingDataset(
        val_df, split="train",
        fixed_height=fixed_height, font_size=font_size, font_path=font_path
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader


def create_test_dataloader(test_csv, batch_size=16, fixed_height=32, font_size=24, font_path=None):
    """Tạo DataLoader cho test set (không có target)."""
    test_df = pd.read_csv(test_csv)
    
    test_ds = PixelUnderstandingDataset(
        test_df, split="test",
        fixed_height=fixed_height, font_size=font_size, font_path=font_path
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    return test_loader