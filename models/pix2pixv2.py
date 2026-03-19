import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CÁC BLOCK CƠ BẢN ĐÃ NÂNG CẤP
# ==========================================
class ResidualBlock(nn.Module):
    """Thay thế DoubleConv bằng Residual Block để chống hụt Gradient"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Đường tắt (Shortcut) để cân bằng số channels nếu in != out
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv_block(x)
        out += self.shortcut(x) # Cộng dồn đường tắt
        return F.leaky_relu(out, 0.2)

class AttentionGate(nn.Module):
    """Cổng Attention giúp mô hình tập trung vào nét chữ"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal từ lớp dưới (nhiều ngữ cảnh hơn)
        # x: skip connection từ encoder (nhiều chi tiết không gian hơn)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi # Áp dụng mặt nạ attention lên x

# ==========================================
# 2. GENERATOR (ATTENTION U-NET + RESIDUAL)
# ==========================================
class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNetGenerator, self).__init__()

        # Encoder dùng Residual Blocks
        self.inc = ResidualBlock(input_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))

        # Attention Gates
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.att1 = AttentionGate(F_g=32, F_l=32, F_int=16)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = ResidualBlock(256, 128) 

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = ResidualBlock(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = ResidualBlock(64, 32)

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
        
        # Áp dụng Attention trước khi nối (concat)
        x3_att = self.att3(g=d1, x=x3)
        d1 = torch.cat([x3_att, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        diffY = x2.size()[2] - d2.size()[2]
        diffX = x2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x2_att = self.att2(g=d2, x=x2)
        d2 = torch.cat([x2_att, d2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        diffY = x1.size()[2] - d3.size()[2]
        diffX = x1.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x1_att = self.att1(g=d3, x=x1)
        d3 = torch.cat([x1_att, d3], dim=1)
        d3 = self.conv_up3(d3)

        return self.outc(d3)

# ==========================================
# 3. DISCRIMINATOR ĐÃ THÊM SPECTRAL NORM
# ==========================================
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=1, mask_channels=1):
        super(PatchGANDiscriminator, self).__init__()
        in_channels = input_channels + mask_channels
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            # THÊM SPECTRAL NORMALIZATION VÀO CONV2D
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1))]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, padding=1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, img, mask):
        model_input = torch.cat([img, mask], dim=1)
        return self.model(model_input)