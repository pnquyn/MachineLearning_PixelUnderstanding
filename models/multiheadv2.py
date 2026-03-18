import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Khối (Conv2d => BatchNorm => LeakyReLU) * 2 để trích xuất đặc trưng"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True), # Dùng LeakyReLU tốt hơn cho các tác vụ sinh ảnh
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetMultiHeadV2(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(UNetMultiHeadV2, self).__init__()

        # ------------------------------------------
        # HEAD 1: U-NET GENERATOR (Sinh ra Text Mask)
        # ------------------------------------------
        # Encoder
        self.inc = DoubleConv(input_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) # Bottleneck

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) 

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        # Output Head 1: Xuất ra LOGITS ảnh 1 kênh (Dự đoán Pixel của Text)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

        # ------------------------------------------
        # HEAD 2: CONDITIONAL CLASSIFIER (Ép sinh Text)
        # ------------------------------------------
        # Nhận vào 2 channels: (Ảnh Input gốc + Ảnh Text sinh ra)
        self.classifier_head = nn.Sequential(
            # Nén lần 1 (H/2, W/2)
            nn.Conv2d(input_channels + 1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Nén lần 2 (H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Nén lần 3 (H/8, W/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Gom về vector đặc trưng
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Phân loại
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # ==========================================
        # PHA 1: SINH ẢNH (GENERATOR)
        # ==========================================
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder với Skip Connections
        d1 = self.up1(x4)
        diffY = x3.size()[2] - d1.size()[2]
        diffX = x3.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([x3, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        diffY = x2.size()[2] - d2.size()[2]
        diffX = x2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        diffY = x1.size()[2] - d3.size()[2]
        diffX = x1.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = torch.cat([x1, d3], dim=1)
        d3 = self.conv_up3(d3)

        # Logits thô của ảnh (chưa qua activation)
        pixel_logits = self.outc(d3) # Shape: (B, 1, H, W)

        # ==========================================
        # PHA 2: KIỂM TRA & PHÂN LOẠI (CLASSIFIER)
        # ==========================================
        # 1. Biến logits thô thành xác suất (Mask từ 0 đến 1) để mạng CNN đọc được
        generated_mask = torch.sigmoid(pixel_logits)

        # 2. Nối ảnh gốc (x) và Mask sinh ra (generated_mask)
        # Mục đích: Ép Head 2 phải đánh giá Mask dựa trên bối cảnh của ảnh gốc
        combined_input = torch.cat([x, generated_mask], dim=1) # Shape: (B, 2, H, W)

        # 3. Phân loại xem cái Mask vừa sinh ra trên nền ảnh gốc có thực sự là Text không
        text_logits = self.classifier_head(combined_input) # Shape: (B, num_classes)

        return pixel_logits, text_logits

# --- Test thử kích thước ---
if __name__ == "__main__":
    model = UNetMultiHeadV2(input_channels=1, num_classes=2)
    dummy_input = torch.randn(2, 1, 256, 256) # Batch=2, Channel=1, 256x256
    
    pixel_out, class_out = model(dummy_input)
    print("Shape ảnh sinh ra (Head 1):", pixel_out.shape) # Output: (2, 1, 256, 256)
    print("Shape phân loại (Head 2):", class_out.shape)   # Output: (2, 2)