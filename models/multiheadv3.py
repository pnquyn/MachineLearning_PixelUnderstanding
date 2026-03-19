import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CÁC BLOCK CƠ BẢN
# ==========================================
class DoubleConv(nn.Module):
    """Khối trích xuất đặc trưng cơ bản cho Generator (U-Net)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ==========================================
# 2. GENERATOR (MẠNG SINH - U-NET)
# ==========================================
class UNetGenerator(nn.Module):
    """
    Nhiệm vụ: Nhận ảnh gốc -> Sinh ra Pixel Mask chứa chữ.
    """
    def __init__(self, input_channels=1, output_channels=1):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.inc = DoubleConv(input_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128) 

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        # Output Head
        self.outc = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

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

        # Trả về LOGITS (chưa qua sigmoid) để tính Dice/BCE Loss cho chuẩn
        return self.outc(d3)

# ==========================================
# 3. DISCRIMINATOR (MẠNG PHÂN BIỆT - PATCH GAN)
# ==========================================
class PatchGANDiscriminator(nn.Module):
    """
    Nhiệm vụ: Nhận vào (Ảnh gốc + Mask). 
    Trả ra 1 ma trận điểm số (Patch) để đánh giá từng vùng nhỏ xem nét chữ là Thật hay Giả.
    """
    def __init__(self, input_channels=1, mask_channels=1):
        super(PatchGANDiscriminator, self).__init__()
        
        # In_channels = Ảnh gốc + Ảnh Mask
        in_channels = input_channels + mask_channels
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Khối Conv -> (BatchNorm) -> LeakyReLU"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Lớp đầu tiên không dùng BatchNorm
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            # Lớp cuối giữ nguyên kích thước (stride=1) để soi chi tiết nét chữ
            nn.Conv2d(256, 512, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output ra một ma trận 2D (Patch), mỗi pixel đại diện cho độ "thật" của 1 vùng ảnh
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, img, mask):
        # Nối ảnh gốc và mask lại với nhau
        model_input = torch.cat([img, mask], dim=1)
        # Trả về Logits của các patch (chưa qua Sigmoid)
        return self.model(model_input)

# --- Test thử kích thước ---
if __name__ == "__main__":
    device = torch.device("cpu")
    
    # Khởi tạo 2 mạng riêng biệt
    generator = UNetGenerator(input_channels=1, output_channels=1).to(device)
    discriminator = PatchGANDiscriminator(input_channels=1, mask_channels=1).to(device)
    
    # Tạo dữ liệu giả: Batch=2, Channel=1, Ảnh 256x256
    dummy_img = torch.randn(2, 1, 256, 256)
    dummy_real_mask = torch.randn(2, 1, 256, 256) # Ground Truth
    
    # 1. Generator sinh ra mask (Logits)
    fake_mask_logits = generator(dummy_img)
    # Áp dụng sigmoid để thành ảnh [0, 1] trước khi đưa cho Discriminator soi
    fake_mask = torch.sigmoid(fake_mask_logits)
    
    # 2. Discriminator đánh giá Mask Giả (do Generator sinh ra)
    pred_fake = discriminator(dummy_img, fake_mask)
    
    # 3. Discriminator đánh giá Mask Thật (Ground Truth)
    pred_real = discriminator(dummy_img, dummy_real_mask)
    
    print("Shape Ảnh gốc:", dummy_img.shape)
    print("Shape Ảnh do G sinh (Fake Mask):", fake_mask.shape)
    
    # Output của D sẽ là một ma trận, ví dụ (2, 1, 15, 15). 
    # Mỗi điểm trong 15x15 đánh giá 1 vùng nhỏ của ảnh gốc xem có phải nét chữ xịn không!
    print("Shape Phân biệt (D chấm điểm):", pred_fake.shape)