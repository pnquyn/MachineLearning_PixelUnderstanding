import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
import torchvision.transforms as T

from tqdm import tqdm

def _pad_width_to_multiple_of_16(w):
    """
    Làm tròn width lên bội số của 16.
    Rất quan trọng cho U-Net để tránh lỗi lệch pixel ở các bước Skip Connection
    khi đi qua các lớp MaxPool2d và ConvTranspose2d.
    """
    return ((w + 15) // 16) * 16

class PixelUnderstandingDataset(Dataset):
    def __init__(self, data, split="train", fixed_height=32, font_size=24, font_path=None):
        """
        data: DataFrame chứa dữ liệu
        split: 'train', 'val' hoặc 'test'
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

        if self.split == "train":
            self.aug = T.Compose([
                # 1. Làm mờ nhẹ (Blur)
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
                
                # 2. Thay đổi độ sáng, tương phản (Jitter)
                T.ColorJitter(brightness=0.3, contrast=0.3),
                
                # 3. Xoay nhẹ (Rotation) - Chỉ xoay tối đa 2 độ để tránh mất nét
                T.RandomRotation(degrees=2, fill=0),
                
                # 4. Dịch chuyển nhẹ (Affine)
                T.RandomAffine(degrees=0, translate=(0.02, 0.02), fill=0)
            ])
        else:
            self.aug = None
        print(f"Loaded {len(self.df)} samples for {split} split. Font: {self.font_path}")

    def _text_to_image_fixed(self, text, max_w):
        """Tạo ảnh grayscale nền đen chữ trắng, ép cứng chiều rộng theo max_w từ file CSV"""
        if pd.isna(text) or not isinstance(text, str):
            text = ""
        
        try:
            font = ImageFont.truetype(self.font_path, self.font_size) if self.font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # 1. TẠO CANVAS CỐ ĐỊNH: Chiều rộng đúng bằng max_width
        img = Image.new('L', (int(max_w), self.fixed_height), color=0)
        draw = ImageDraw.Draw(img)
        
        mask = font.getmask(text)
        _, text_height = mask.size
        
        # Căn giữa dọc
        y_offset = max((self.fixed_height - text_height) // 2, 0)
        draw.text((5, y_offset), text, fill=255, font=font)
        
        return img

    def __len__(self):
        return len(self.df)

    def _calculate_pixel_width(self, char_count):
        # Dùng 0.5 hoặc 0.6 là mức tối thiểu an toàn cho Times New Roman 24
        pixel_w = int(char_count * (self.font_size * 0.4)) + 10 
        
        # Làm tròn lên bội số của 16 ngay tại đây giúp giảm việc padding thừa trong model
        return ((pixel_w + 5) // 16) * 16

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        
        # 1. CHUYỂN ĐỔI: char_count (từ CSV) -> pixel_width
        char_max = row['max_width']
        pixel_max_w = self._calculate_pixel_width(char_max)
        
        # 2. XỬ LÝ CONTEXT (Input)
        # Sử dụng pixel_max_w vừa tính được thay vì char_max
        context_img = self._text_to_image_fixed(row['context'], pixel_max_w)
        if self.aug:
            # torchvision transforms hoạt động trực tiếp trên PIL Image
            context_img = self.aug(context_img)
        
        context_tensor = F.to_tensor(context_img)
        
        # 3. XỬ LÝ TARGET (Dành cho train và val)
        if self.split in ["train", "val"]:
            target_text = row['target'] if 'target' in row else ""
            
            # Ảnh target (Head 1)
            target_img = self._text_to_image_fixed(target_text, pixel_max_w)
            target_tensor = F.to_tensor(target_img)
            
            # Nhãn phân loại (Head 2)
            is_valid_text = 1 if isinstance(target_text, str) and len(target_text.strip()) > 0 else 0
            target_label = torch.tensor(is_valid_text, dtype=torch.long)
            
            return context_tensor, target_tensor, target_label, torch.tensor(sample_id, dtype=torch.long)

        # 4. XỬ LÝ TEST
        return context_tensor, torch.tensor(sample_id, dtype=torch.long)


def collate_fn(batch):
    """
    Gom Batch linh hoạt: Tìm width lớn nhất trong Batch hiện tại,
    pad bằng nền đen (0) để các ảnh cùng kích thước.
    """
    if len(batch[0]) == 4:  # Train/Val mode
        contexts, targets, labels, ids = zip(*batch)
        
        # Lấy width lớn nhất trong lô này (dựa trên max_width đã gen ra)
        batch_max_w = max(img.shape[2] for img in contexts)
        batch_max_w = _pad_width_to_multiple_of_16(batch_max_w)
        
        # Pad context và target bằng hàm F.pad (padding bên phải)
        padded_contexts = torch.stack([
            F.pad(img, (0, 0, batch_max_w - img.shape[2], 0), fill=0) for img in contexts
        ])
        padded_targets = torch.stack([
            F.pad(img, (0, 0, batch_max_w - img.shape[2], 0), fill=0) for img in targets
        ])
        
        return padded_contexts, padded_targets, torch.stack(labels), torch.stack(ids)
    
    else:  # Test mode
        contexts, ids = zip(*batch)
        
        batch_max_w = max(img.shape[2] for img in contexts)
        batch_max_w = _pad_width_to_multiple_of_16(batch_max_w)
        
        padded_contexts = torch.stack([
            F.pad(img, (0, 0, batch_max_w - img.shape[2], 0), fill=0) for img in contexts
        ])
        
        return padded_contexts, torch.stack(ids)


def create_dataloaders(train_csv, val_csv, batch_size=16, fixed_height=32, font_size=24, font_path=None):
    # 1. Đọc dữ liệu từ hai file riêng biệt
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # In thông tin kiểm tra
    print(f"--- Data Loaded ---")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    
    # Kiểm tra nhanh phân phối ngôn ngữ (optional)
    if 'language' in train_df.columns:
        print(f"Languages in Train: {train_df['language'].nunique()}")
        print(f"Languages in Val:   {val_df['language'].nunique()}")

    # 2. Khởi tạo Dataset
    train_ds = PixelUnderstandingDataset(
        train_df, 
        split="train",
        fixed_height=fixed_height, 
        font_size=font_size, 
        font_path=font_path
    )
    
    val_ds = PixelUnderstandingDataset(
        val_df, 
        split="val",
        fixed_height=fixed_height, 
        font_size=font_size, 
        font_path=font_path
    )

    # 3. Khởi tạo DataLoader
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, # Train thường cần shuffle để model generalize tốt hơn
        collate_fn=collate_fn, 
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, # Val thường không cần shuffle để dễ track kết quả
        collate_fn=collate_fn, 
        num_workers=0, 
        pin_memory=True
    )

    return train_loader, val_loader


def create_test_dataloader(test_csv, batch_size=16, fixed_height=32, font_size=24, font_path=None):
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

def export_preprocess_samples(csv_path, output_dir="check_preprocess", num_samples=20):
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file {csv_path}")
        return

    # 1. Khởi tạo Dataset (Dùng split='train' để lấy được cả target)
    df = pd.read_csv(csv_path)
    dataset = PixelUnderstandingDataset(
        df, 
        split="train", 
        fixed_height=32, 
        font_size=24
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Đang test __getitem__ với {num_samples} mẫu ---")

    # Lấy mẫu ngẫu nhiên hoặc n mẫu đầu tiên
    for i in tqdm(range(min(num_samples, len(dataset)))):
        # Gọi trực tiếp __getitem__
        # Kết quả trả về: context_tensor, target_tensor, target_label, sample_id
        context_tensor, target_tensor, target_label, sample_id = dataset[i]
        
        sid = sample_id.item()
        
        # 2. Chuyển Tensor ngược lại thành Image để lưu
        # F.to_pil_image sẽ chuyển Tensor (1, H, W) về ảnh PIL
        img_ctx = F.to_pil_image(context_tensor)
        img_tgt = F.to_pil_image(target_tensor)
        
        # Lấy kích thước ảnh thực tế sau khi đã tính từ char_count
        actual_w = img_ctx.width
        h = img_ctx.height
        
        # 3. Ghép đôi để so sánh (Context trên, Target dưới)
        combined = Image.new('L', (actual_w, h * 2 + 5), color=128)
        combined.paste(img_ctx, (0, 0))
        combined.paste(img_tgt, (0, h + 5))
        
        # Lưu ảnh
        save_path = os.path.join(output_dir, f"sample_{sid}_label{target_label.item()}.png")
        combined.save(save_path)

    print(f"\nHoàn tất! Kiểm tra thư mục: '{output_dir}'")
    print(f"Ghi chú: Tên file có dạng sample_[ID]_label[0/1].png")

if __name__ == "__main__":
    # Thay đường dẫn này bằng file val của bạn
    export_preprocess_samples("data/val_split.csv")