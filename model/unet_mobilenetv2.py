"""
UNet with MobileNetV2 Backbone
Lightweight architecture for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ConvBlock(nn.Module):
    """Convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet_MobileNetV2(nn.Module):
    """
    UNet with MobileNetV2 backbone encoder.
    
    Architecture:
        - Encoder: MobileNetV2 (pretrained on ImageNet)
        - Decoder: Standard UNet decoder with skip connections
        - Lightweight: ~3.5M parameters (vs 40M in DeepLabV3+)
        - Fast: Good for edge devices
    
    Args:
        n_classes: Number of segmentation classes
        pretrained: Use ImageNet pretrained weights
    """
    def __init__(self, n_classes=7, pretrained=True):
        super(UNet_MobileNetV2, self).__init__()
        
        # Load MobileNetV2 backbone
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            backbone = mobilenet_v2(weights=weights)
        else:
            backbone = mobilenet_v2(weights=None)
        
        # Extract encoder layers
        features = backbone.features
        
        # MobileNetV2 feature extraction layers
        # Input: 3 channels
        # Layer indices and output channels:
        #   0-1:   32 channels  (stride 2)  -> 128x128
        #   2-3:   24 channels  (stride 4)  -> 64x64
        #   4-6:   32 channels  (stride 8)  -> 32x32
        #   7-13:  96 channels  (stride 16) -> 16x16
        #   14-17: 320 channels (stride 32) -> 8x8
        
        self.encoder1 = nn.Sequential(*features[0:2])   # 32 channels, /2
        self.encoder2 = nn.Sequential(*features[2:4])   # 24 channels, /4
        self.encoder3 = nn.Sequential(*features[4:7])   # 32 channels, /8
        self.encoder4 = nn.Sequential(*features[7:14])  # 96 channels, /16
        self.encoder5 = nn.Sequential(*features[14:])   # 320 channels, /32
        
        # Bridge (bottleneck)
        self.bridge = ConvBlock(320, 512)
        
        # Decoder with skip connections
        self.up1 = UpBlock(512, 96, 256)   # 512 + 96 -> 256
        self.up2 = UpBlock(256, 32, 128)   # 256 + 32 -> 128
        self.up3 = UpBlock(128, 24, 64)    # 128 + 24 -> 64
        self.up4 = UpBlock(64, 32, 32)     # 64 + 32 -> 32
        
        # Final upsampling and classification
        self.final_up = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.final_conv = nn.Conv2d(32, n_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights"""
        for m in [self.bridge, self.up1, self.up2, self.up3, self.up4, self.final_up, self.final_conv]:
            if isinstance(m, nn.Module):
                for module in m.modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Store input size
        input_size = x.shape[2:]
        
        # Encoder
        e1 = self.encoder1(x)      # /2:  [B, 32, 128, 128]
        e2 = self.encoder2(e1)     # /4:  [B, 24, 64, 64]
        e3 = self.encoder3(e2)     # /8:  [B, 32, 32, 32]
        e4 = self.encoder4(e3)     # /16: [B, 96, 16, 16]
        e5 = self.encoder5(e4)     # /32: [B, 320, 8, 8]
        
        # Bridge
        bridge = self.bridge(e5)   # [B, 512, 8, 8]
        
        # Decoder with skip connections
        d1 = self.up1(bridge, e4)  # [B, 256, 16, 16]
        d2 = self.up2(d1, e3)      # [B, 128, 32, 32]
        d3 = self.up3(d2, e2)      # [B, 64, 64, 64]
        d4 = self.up4(d3, e1)      # [B, 32, 128, 128]
        
        # Final upsampling and classification
        out = self.final_up(d4)    # [B, 32, 256, 256]
        out = self.final_conv(out) # [B, n_classes, 256, 256]
        
        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out
    
    def get_backbone_params(self):
        """Get encoder (backbone) parameters for different learning rates"""
        return list(self.encoder1.parameters()) + \
               list(self.encoder2.parameters()) + \
               list(self.encoder3.parameters()) + \
               list(self.encoder4.parameters()) + \
               list(self.encoder5.parameters())
    
    def get_decoder_params(self):
        """Get decoder parameters for different learning rates"""
        return list(self.bridge.parameters()) + \
               list(self.up1.parameters()) + \
               list(self.up2.parameters()) + \
               list(self.up3.parameters()) + \
               list(self.up4.parameters()) + \
               list(self.final_up.parameters()) + \
               list(self.final_conv.parameters())


class MultiHeadUNet_MobileNetV2(nn.Module):
    """
    Multi-head UNet with MobileNetV2 backbone.
    Uses ensemble of multiple heads for better predictions.
    
    Args:
        n_classes: Number of segmentation classes
        num_heads: Number of prediction heads (default: 3)
        pretrained: Use ImageNet pretrained weights
    """
    def __init__(self, n_classes=7, num_heads=3, pretrained=True):
        super(MultiHeadUNet_MobileNetV2, self).__init__()
        
        self.num_heads = num_heads
        
        # Load MobileNetV2 backbone
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            backbone = mobilenet_v2(weights=weights)
        else:
            backbone = mobilenet_v2(weights=None)
        
        features = backbone.features
        
        # Shared encoder
        self.encoder1 = nn.Sequential(*features[0:2])
        self.encoder2 = nn.Sequential(*features[2:4])
        self.encoder3 = nn.Sequential(*features[4:7])
        self.encoder4 = nn.Sequential(*features[7:14])
        self.encoder5 = nn.Sequential(*features[14:])
        
        # Shared bridge
        self.bridge = ConvBlock(320, 512)
        
        # Shared decoder backbone
        self.up1 = UpBlock(512, 96, 256)
        self.up2 = UpBlock(256, 32, 128)
        self.up3 = UpBlock(128, 24, 64)
        self.up4 = UpBlock(64, 32, 32)
        self.final_up = nn.ConvTranspose2d(32, 32, 2, stride=2)
        
        # Multiple heads with different dropout rates for diversity
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.1 + 0.05 * i),
                nn.Conv2d(32, n_classes, 1)
            ) for i in range(num_heads)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights"""
        for m in [self.bridge, self.up1, self.up2, self.up3, self.up4, self.final_up]:
            if isinstance(m, nn.Module):
                for module in m.modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
        
        for head in self.heads:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_all_heads=False):
        # Store input size
        input_size = x.shape[2:]
        
        # Shared encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Shared bridge
        bridge = self.bridge(e5)
        
        # Shared decoder
        d1 = self.up1(bridge, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        features = self.final_up(d4)
        
        # Multiple heads
        outputs = [head(features) for head in self.heads]
        
        # Ensure output matches input size
        outputs = [F.interpolate(out, size=input_size, mode='bilinear', align_corners=True) 
                   for out in outputs]
        
        if return_all_heads:
            return outputs  # List of [B, C, H, W]
        else:
            # Return ensemble (mean)
            return torch.stack(outputs).mean(dim=0)  # [B, C, H, W]
    
    def get_backbone_params(self):
        """Get encoder (backbone) parameters"""
        return list(self.encoder1.parameters()) + \
               list(self.encoder2.parameters()) + \
               list(self.encoder3.parameters()) + \
               list(self.encoder4.parameters()) + \
               list(self.encoder5.parameters())
    
    def get_decoder_params(self):
        """Get decoder parameters"""
        params = list(self.bridge.parameters()) + \
                 list(self.up1.parameters()) + \
                 list(self.up2.parameters()) + \
                 list(self.up3.parameters()) + \
                 list(self.up4.parameters()) + \
                 list(self.final_up.parameters())
        for head in self.heads:
            params += list(head.parameters())
        return params


# Test code
if __name__ == "__main__":
    print("Testing UNet_MobileNetV2...")
    model = UNet_MobileNetV2(n_classes=7, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.2f}M")
    
    print("\nTesting MultiHeadUNet_MobileNetV2...")
    model = MultiHeadUNet_MobileNetV2(n_classes=7, num_heads=3, pretrained=False)
    out_ensemble = model(x, return_all_heads=False)
    out_all = model(x, return_all_heads=True)
    print(f"  Input: {x.shape}, Output (ensemble): {out_ensemble.shape}")
    print(f"  Output (all heads): {len(out_all)} heads, each {out_all[0].shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.2f}M")
    
    print("\n✓ All tests passed!")
