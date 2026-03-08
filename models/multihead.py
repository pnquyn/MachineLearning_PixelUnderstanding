import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Khối (Convolution => BatchNorm => ReLU) * 2 để trích xuất đặc trưng"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetMultiHeadModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(UNetMultiHeadModel, self).__init__()

        # ==========================================
        # 1. ENCODER (Nhánh xuống - Downsampling)
        # ==========================================
        self.inc = DoubleConv(input_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) # Bottleneck

        # ==========================================
        # 2. HEAD 1: PIXEL GENERATOR (Nhánh lên - Decoder)
        # ==========================================
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) 

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        # Output Head 1: Xuất ra LOGITS 
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

        # ==========================================
        # 3. HEAD 2: CLASSIFIER (Text vs Non-Text)
        # ==========================================
        # AdaptiveAvgPool2d đưa mọi kích thước (H, W) của Bottleneck về (1, 1)
        self.classifier_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Output Head 2: Xuất ra LOGITS 
            nn.Linear(128, num_classes) 
        )

    def forward(self, x):
        # --- ENCODER ---
        x1 = self.inc(x)        # Cấp 1
        x2 = self.down1(x1)     # Cấp 2
        x3 = self.down2(x2)     # Cấp 3
        x4 = self.down3(x3)     # Bottleneck (Đặc trưng nén sâu nhất)

        # --- HEAD 1: TÁI TẠO PIXEL (DECODER + SKIP CONNECTIONS) ---
        d1 = self.up1(x4)
        # F.pad giúp căn chỉnh kích thước nếu H, W không chia hết chẵn cho 2
        diffY = x3.size()[2] - d1.size()[2]
        diffX = x3.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Skip connection: Nối đặc trưng không gian từ x3 sang d1
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

        pixel_logits = self.outc(d3) # Shape: (B, 1, H, W) khớp 100% với Input x

        # --- HEAD 2: PHÂN LOẠI TEXT/NON-TEXT ---
        # Lấy đặc trưng sâu nhất từ Bottleneck (x4)
        cls_feat = self.classifier_pool(x4)
        text_logits = self.classifier_head(cls_feat) # Shape: (B, 2)

        return pixel_logits, text_logits