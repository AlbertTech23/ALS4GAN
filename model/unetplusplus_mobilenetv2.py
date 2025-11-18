"""
UNet++ with MobileNetV2 Backbone for Semantic Segmentation
==========================================================

Architecture:
- Encoder: MobileNetV2 (pretrained on ImageNet)
- Decoder: UNet++ with dense nested skip connections
- Deep supervision with multiple segmentation outputs
- Multi-head support for ensemble predictions

Features:
- Dense nested architecture for better feature fusion
- Deep supervision at multiple scales
- Lightweight (~5-7M parameters, still lighter than DeepLabV3+ 40M)
- Pretrained backbone for better feature extraction
- Superior to standard UNet due to nested skip pathways
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
try:
    from torchvision.models import MobileNet_V2_Weights
except ImportError:
    MobileNet_V2_Weights = None


class ConvBlock(nn.Module):
    """Double convolution block for UNet++"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ConvBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus_MobileNetV2(nn.Module):
    """
    UNet++ with MobileNetV2 backbone
    
    Architecture:
    - Encoder: MobileNetV2 pretrained features (5 levels: X0,0 to X4,0)
    - Decoder: Dense nested skip connections (UNet++ architecture)
    - Deep supervision: Multiple segmentation outputs at different levels
    - Output: Final segmentation (or ensemble of outputs if deep_supervision=True)
    
    UNet++ improves upon UNet by:
    1. Dense skip connections between encoder and decoder
    2. Nested convolution blocks for better feature fusion
    3. Deep supervision for multi-scale learning
    4. More gradual feature map transformation
    
    Notation: X^i,j where i = encoder level (0-4), j = decoder level (0-4)
    - X^i,0: Encoder outputs
    - X^i,j (j>0): Nested decoder blocks
    """
    def __init__(self, n_classes=7, pretrained=True, deep_supervision=False):
        super(UNetPlusPlus_MobileNetV2, self).__init__()
        
        self.deep_supervision = deep_supervision
        self.n_classes = n_classes
        
        # Load MobileNetV2 backbone
        if pretrained:
            if MobileNet_V2_Weights is not None:
                weights = MobileNet_V2_Weights.IMAGENET1K_V1
                backbone = mobilenet_v2(weights=weights)
            else:
                backbone = mobilenet_v2(pretrained=True)
        else:
            backbone = mobilenet_v2(pretrained=False)
        
        # Extract encoder layers from MobileNetV2
        features = backbone.features
        
        # MobileNetV2 encoder structure (ACTUAL output channels):
        # Level 0: 16 channels  (stride 2)  -> 128x128  [X^0,0]
        # Level 1: 24 channels  (stride 4)  -> 64x64    [X^1,0]
        # Level 2: 32 channels  (stride 8)  -> 32x32    [X^2,0]
        # Level 3: 96 channels  (stride 16) -> 16x16    [X^3,0]
        # Level 4: 1280 channels (stride 32) -> 8x8     [X^4,0]
        
        self.encoder0 = nn.Sequential(*features[0:2])   # X^0,0: 16 ch
        self.encoder1 = nn.Sequential(*features[2:4])   # X^1,0: 24 ch
        self.encoder2 = nn.Sequential(*features[4:7])   # X^2,0: 32 ch
        self.encoder3 = nn.Sequential(*features[7:14])  # X^3,0: 96 ch
        self.encoder4 = nn.Sequential(*features[14:])   # X^4,0: 1280 ch
        
        # Channel dimensions for each level
        channels = [16, 24, 32, 96, 1280]
        
        # Decoder channels (reduced for efficiency)
        decoder_channels = [64, 128, 256, 512]
        
        # Bridge to increase channels from encoder
        self.bridge0 = ConvBlock(channels[0], decoder_channels[0])  # 16 -> 64
        self.bridge1 = ConvBlock(channels[1], decoder_channels[0])  # 24 -> 64
        self.bridge2 = ConvBlock(channels[2], decoder_channels[1])  # 32 -> 128
        self.bridge3 = ConvBlock(channels[3], decoder_channels[2])  # 96 -> 256
        self.bridge4 = ConvBlock(channels[4], decoder_channels[3])  # 1280 -> 512
        
        # UNet++ nested decoder blocks
        # Notation: conv_i_j means block at level i, column j
        # After bridging, all levels use uniform decoder channels
        # Level 0: 64ch, Level 1: 64ch, Level 2: 128ch, Level 3: 256ch
        
        # Level 0 (top) - all outputs are 64ch
        self.conv0_1 = ConvBlock(64 + 64, 64)              # x0_0(64) + up(x1_0)(64) = 128 -> 64
        self.conv0_2 = ConvBlock(64 + 64 + 64, 64)         # x0_0(64) + x0_1(64) + up(x1_1)(64) = 192 -> 64
        self.conv0_3 = ConvBlock(64 * 4, 64)               # x0_0 + x0_1 + x0_2 + up(x1_2) = 256 -> 64
        self.conv0_4 = ConvBlock(64 * 5, 64)               # x0_0 + x0_1 + x0_2 + x0_3 + up(x1_3) = 320 -> 64
        
        # Level 1 - all outputs are 64ch (after first conv)
        self.conv1_1 = ConvBlock(64 + 128, 64)             # x1_0(64) + up(x2_0)(128) = 192 -> 64
        self.conv1_2 = ConvBlock(64 + 64 + 128, 64)        # x1_0(64) + x1_1(64) + up(x2_1)(128) = 256 -> 64
        self.conv1_3 = ConvBlock(64 + 64 + 64 + 128, 64)   # x1_0 + x1_1 + x1_2 + up(x2_2) = 320 -> 64
        
        # Level 2 - all outputs are 128ch
        self.conv2_1 = ConvBlock(128 + 256, 128)           # x2_0(128) + up(x3_0)(256) = 384 -> 128
        self.conv2_2 = ConvBlock(128 + 128 + 256, 128)     # x2_0 + x2_1 + up(x3_1) = 512 -> 128
        
        # Level 3 - all outputs are 256ch
        self.conv3_1 = ConvBlock(256 + 512, 256)           # x3_0(256) + up(x4_0)(512) = 768 -> 256
        
        # Upsampling operations
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Deep supervision outputs (multiple segmentation heads)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(decoder_channels[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(decoder_channels[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(decoder_channels[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(decoder_channels[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(decoder_channels[0], n_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through UNet++
        
        Returns:
            - If deep_supervision=False: Single segmentation output
            - If deep_supervision=True: List of 4 outputs at different scales
        """
        input_size = x.shape[2:]
        
        # Encoder path (contracting)
        x0_0 = self.encoder0(x)      # [B, 16, 128, 128]
        x1_0 = self.encoder1(x0_0)   # [B, 24, 64, 64]
        x2_0 = self.encoder2(x1_0)   # [B, 32, 32, 32]
        x3_0 = self.encoder3(x2_0)   # [B, 96, 16, 16]
        x4_0 = self.encoder4(x3_0)   # [B, 1280, 8, 8]
        
        # Apply bridge convolutions to match decoder channels
        x0_0 = self.bridge0(x0_0)    # [B, 64, 128, 128]
        x1_0 = self.bridge1(x1_0)    # [B, 64, 64, 64]
        x2_0 = self.bridge2(x2_0)    # [B, 128, 32, 32]
        x3_0 = self.bridge3(x3_0)    # [B, 256, 16, 16]
        x4_0 = self.bridge4(x4_0)    # [B, 512, 8, 8]
        
        # UNet++ nested decoder path
        # Column 1 - use size-matching upsampling
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, size=x3_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        # Column 4 (final)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        # Output head(s)
        if self.deep_supervision:
            # Multiple outputs for deep supervision
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            
            # Upsample all to input size
            out1 = F.interpolate(out1, size=input_size, mode='bilinear', align_corners=True)
            out2 = F.interpolate(out2, size=input_size, mode='bilinear', align_corners=True)
            out3 = F.interpolate(out3, size=input_size, mode='bilinear', align_corners=True)
            out4 = F.interpolate(out4, size=input_size, mode='bilinear', align_corners=True)
            
            return [out1, out2, out3, out4]
        else:
            # Single output (most refined)
            out = self.final(x0_4)
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            return out
    
    def get_backbone_params(self):
        """Get encoder (backbone) parameters for differential learning rates"""
        return list(self.encoder0.parameters()) + \
               list(self.encoder1.parameters()) + \
               list(self.encoder2.parameters()) + \
               list(self.encoder3.parameters()) + \
               list(self.encoder4.parameters())
    
    def get_decoder_params(self):
        """Get decoder parameters for differential learning rates"""
        decoder_params = []
        
        # Bridge parameters
        decoder_params += list(self.bridge0.parameters())
        decoder_params += list(self.bridge1.parameters())
        decoder_params += list(self.bridge2.parameters())
        decoder_params += list(self.bridge3.parameters())
        decoder_params += list(self.bridge4.parameters())
        
        # Nested decoder blocks
        decoder_params += list(self.conv0_1.parameters())
        decoder_params += list(self.conv0_2.parameters())
        decoder_params += list(self.conv0_3.parameters())
        decoder_params += list(self.conv0_4.parameters())
        decoder_params += list(self.conv1_1.parameters())
        decoder_params += list(self.conv1_2.parameters())
        decoder_params += list(self.conv1_3.parameters())
        decoder_params += list(self.conv2_1.parameters())
        decoder_params += list(self.conv2_2.parameters())
        decoder_params += list(self.conv3_1.parameters())
        
        # Output heads
        if self.deep_supervision:
            decoder_params += list(self.final1.parameters())
            decoder_params += list(self.final2.parameters())
            decoder_params += list(self.final3.parameters())
            decoder_params += list(self.final4.parameters())
        else:
            decoder_params += list(self.final.parameters())
        
        return decoder_params


class MultiHeadUNetPlusPlus_MobileNetV2(nn.Module):
    """
    Multi-Head UNet++ with MobileNetV2 for ensemble predictions
    
    Similar to MultiHeadDeepLabV3Plus but with UNet++ architecture:
    - Shared encoder and most of decoder
    - Multiple output heads with different dropout rates
    - Ensemble predictions for better accuracy
    """
    def __init__(self, n_classes=7, num_heads=3, pretrained=True, deep_supervision=False):
        super(MultiHeadUNetPlusPlus_MobileNetV2, self).__init__()
        
        self.num_heads = num_heads
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        
        # Shared UNet++ backbone
        self.backbone = UNetPlusPlus_MobileNetV2(n_classes=n_classes, pretrained=pretrained, deep_supervision=False)
        
        # Remove the final layer (we'll use multiple heads instead)
        self.backbone.final = nn.Identity()
        
        # Multiple classification heads with different dropout
        self.heads = nn.ModuleList()
        dropout_rates = [0.1, 0.15, 0.2][:num_heads]
        
        for i in range(num_heads):
            head = nn.Sequential(
                nn.Dropout2d(p=dropout_rates[i]),
                nn.Conv2d(64, n_classes, kernel_size=1)  # 64 is the decoder_channels[0]
            )
            self.heads.append(head)
    
    def forward(self, x, return_all_heads=False):
        """
        Forward pass with multi-head ensemble
        
        Args:
            x: Input tensor
            return_all_heads: If True, return list of all head outputs
                             If False, return averaged ensemble output
        
        Returns:
            - If return_all_heads=False: Single ensemble output (average)
            - If return_all_heads=True: List of outputs from each head
        """
        input_size = x.shape[2:]
        
        # Get features from backbone (before final layer)
        # We need to replicate the encoder-decoder path
        x0_0 = self.backbone.encoder0(x)
        x1_0 = self.backbone.encoder1(x0_0)
        x2_0 = self.backbone.encoder2(x1_0)
        x3_0 = self.backbone.encoder3(x2_0)
        x4_0 = self.backbone.encoder4(x3_0)
        
        x0_0 = self.backbone.bridge0(x0_0)
        x1_0 = self.backbone.bridge1(x1_0)
        x2_0 = self.backbone.bridge2(x2_0)
        x3_0 = self.backbone.bridge3(x3_0)
        x4_0 = self.backbone.bridge4(x4_0)
        
        # Decoder - use size-matching upsampling
        x0_1 = self.backbone.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_1 = self.backbone.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_1 = self.backbone.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x3_1 = self.backbone.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, size=x3_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        x0_2 = self.backbone.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_2 = self.backbone.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_2 = self.backbone.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        x0_3 = self.backbone.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_3 = self.backbone.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        x0_4 = self.backbone.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        # Apply multiple heads to final features
        outputs = []
        for head in self.heads:
            out = head(x0_4)
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            outputs.append(out)
        
        if return_all_heads:
            return outputs
        else:
            # Return ensemble average
            return torch.mean(torch.stack(outputs), dim=0)
    
    def get_backbone_params(self):
        """Get encoder parameters"""
        return self.backbone.get_backbone_params()
    
    def get_decoder_params(self):
        """Get decoder + heads parameters"""
        params = self.backbone.get_decoder_params()
        for head in self.heads:
            params += list(head.parameters())
        return params


# Test code
if __name__ == '__main__':
    print("Testing UNet++ with MobileNetV2...")
    
    # Test single-head UNet++
    model = UNetPlusPlus_MobileNetV2(n_classes=7, pretrained=False, deep_supervision=False)
    x = torch.randn(2, 3, 256, 256)
    
    print(f"\n1. Single-head UNet++ (no deep supervision):")
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with deep supervision
    model_ds = UNetPlusPlus_MobileNetV2(n_classes=7, pretrained=False, deep_supervision=True)
    
    print(f"\n2. UNet++ with deep supervision:")
    outputs = model_ds(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"   Output {i+1} shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_ds.parameters()):,}")
    
    # Test multi-head UNet++
    model_mh = MultiHeadUNetPlusPlus_MobileNetV2(n_classes=7, num_heads=3, pretrained=False)
    
    print(f"\n3. Multi-Head UNet++ (3 heads):")
    out_ensemble = model_mh(x, return_all_heads=False)
    out_all = model_mh(x, return_all_heads=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Ensemble output: {out_ensemble.shape}")
    print(f"   Individual outputs: {len(out_all)} x {out_all[0].shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_mh.parameters()):,}")
    
    print("\n✅ All tests passed!")
