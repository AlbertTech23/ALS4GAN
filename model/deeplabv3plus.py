"""
DeepLabV3+ Implementation for Semantic Segmentation
Based on: "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
https://arxiv.org/abs/1802.02611

Key improvements over DeepLabV2:
1. Encoder-Decoder architecture (vs encoder-only)
2. Improved ASPP with separable convolutions
3. Low-level feature fusion in decoder
4. Better performance on fine details
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPPConv(nn.Module):
    """Atrous Convolution with Batch Normalization and ReLU"""
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPooling(nn.Module):
    """Global Average Pooling branch"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.gap(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP++) Module"""
    def __init__(self, in_channels, out_channels=256, output_stride=16):
        super(ASPP, self).__init__()
        
        # Determine dilation rates based on output stride
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError(f"output_stride {output_stride} not supported")
        
        # ASPP branches
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = ASPPConv(in_channels, out_channels, dilations[1])
        self.conv3 = ASPPConv(in_channels, out_channels, dilations[2])
        self.conv4 = ASPPConv(in_channels, out_channels, dilations[3])
        self.pool = ASPPPooling(in_channels, out_channels)
        
        # Projection after concatenation
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        res.append(self.conv1(x))
        res.append(self.conv2(x))
        res.append(self.conv3(x))
        res.append(self.conv4(x))
        res.append(self.pool(x))
        
        res = torch.cat(res, dim=1)
        return self.project(res)


class Decoder(nn.Module):
    """DeepLabV3+ Decoder with low-level feature fusion"""
    def __init__(self, num_classes, low_level_channels=256):
        super(Decoder, self).__init__()
        
        # Low-level feature projection
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutions
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 256 (ASPP) + 48 (low-level) = 304
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # Final classification layer
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, low_level_features):
        # Process low-level features
        low_level = self.low_level_conv(low_level_features)
        
        # Upsample high-level features to match low-level resolution
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate
        x = torch.cat([x, low_level], dim=1)
        
        # Decoder
        x = self.decoder(x)
        x = self.classifier(x)
        
        return x


class DeepLabV3Plus_ResNet50(nn.Module):
    """
    DeepLabV3+ with ResNet50 backbone
    
    Args:
        n_classes: Number of output classes
        output_stride: 8 or 16 (controls resolution of output features)
        pretrained_backbone: Whether to use ImageNet pretrained weights
    """
    def __init__(self, n_classes, output_stride=16, pretrained_backbone=True):
        super(DeepLabV3Plus_ResNet50, self).__init__()
        
        self.n_classes = n_classes
        self.output_stride = output_stride
        
        # Load ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained_backbone)
        
        # Modify ResNet for different output strides
        if output_stride == 16:
            # Standard: stride=2 in layer3, stride=2 in layer4
            self.layer0 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
            )
            self.layer1 = resnet.layer1  # 256 channels, stride 4
            self.layer2 = resnet.layer2  # 512 channels, stride 8
            self.layer3 = resnet.layer3  # 1024 channels, stride 16
            self.layer4 = resnet.layer4  # 2048 channels, stride 16 (modified)
            
            # Replace stride with dilation in layer4
            self._make_layer_dilated(self.layer4, dilation=2, stride=1)
            
        elif output_stride == 8:
            # Higher resolution: stride=1 in layer3, stride=1 in layer4
            self.layer0 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
            )
            self.layer1 = resnet.layer1  # 256 channels, stride 4
            self.layer2 = resnet.layer2  # 512 channels, stride 8
            self.layer3 = resnet.layer3  # 1024 channels, stride 8 (modified)
            self.layer4 = resnet.layer4  # 2048 channels, stride 8 (modified)
            
            # Replace stride with dilation
            self._make_layer_dilated(self.layer3, dilation=2, stride=1)
            self._make_layer_dilated(self.layer4, dilation=4, stride=1)
        
        # ASPP module
        self.aspp = ASPP(in_channels=2048, out_channels=256, output_stride=output_stride)
        
        # Decoder (uses layer1 features as low-level)
        self.decoder = Decoder(num_classes=n_classes, low_level_channels=256)
        
        # Initialize decoder weights
        self._init_decoder_weights()
    
    def _make_layer_dilated(self, layer, dilation, stride=1):
        """Convert layer to use dilated convolutions instead of strided"""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (stride, stride)
                    module.dilation = (dilation, dilation)
                    # Adjust padding to maintain spatial dimensions
                    module.padding = (dilation, dilation)
    
    def _init_decoder_weights(self):
        """Initialize decoder weights with Xavier"""
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Encoder
        x = self.layer0(x)
        low_level_features = self.layer1(x)  # For decoder skip connection
        x = self.layer2(low_level_features)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x, low_level_features)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x
    
    def get_backbone_params(self):
        """Get backbone parameters for different learning rates"""
        return [
            self.layer0.parameters(),
            self.layer1.parameters(),
            self.layer2.parameters(),
            self.layer3.parameters(),
            self.layer4.parameters(),
        ]
    
    def get_decoder_params(self):
        """Get decoder parameters for different learning rates"""
        return [
            self.aspp.parameters(),
            self.decoder.parameters(),
        ]


class MultiHeadDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with multiple classification heads for diversity
    Inspired by DiverseNet's multi-head approach
    """
    def __init__(self, n_classes, num_heads=3, output_stride=16, pretrained_backbone=True):
        super(MultiHeadDeepLabV3Plus, self).__init__()
        
        self.n_classes = n_classes
        self.num_heads = num_heads
        
        # Shared backbone and ASPP
        resnet = models.resnet50(pretrained=pretrained_backbone)
        
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        if output_stride == 16:
            self._make_layer_dilated(self.layer4, dilation=2, stride=1)
        
        self.aspp = ASPP(in_channels=2048, out_channels=256, output_stride=output_stride)
        
        # Low-level feature projection (shared)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Shared decoder backbone
        self.decoder_backbone = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Multiple classification heads with diversity
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.1 + 0.05 * i),  # Different dropout rates for diversity
                nn.Conv2d(256, n_classes, 1)
            ) for i in range(num_heads)
        ])
        
        self._init_weights()
    
    def _make_layer_dilated(self, layer, dilation, stride=1):
        """Convert layer to use dilated convolutions"""
        # Modify the first bottleneck's conv2 (3x3 conv)
        for i, block in enumerate(layer):
            # Fix the 3x3 conv in each bottleneck
            if hasattr(block, 'conv2'):
                if block.conv2.stride == (2, 2) and i == 0:
                    # First block: change stride to 1, add dilation
                    block.conv2.stride = (stride, stride)
                    block.conv2.dilation = (dilation, dilation)
                    block.conv2.padding = (dilation, dilation)
                    
                    # Also fix the downsampling layer
                    if block.downsample is not None:
                        block.downsample[0].stride = (stride, stride)
                else:
                    # Other blocks: just add dilation
                    block.conv2.dilation = (dilation, dilation)
                    block.conv2.padding = (dilation, dilation)
    
    def _init_weights(self):
        """Initialize decoder weights"""
        for m in self.decoder_backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for head in self.heads:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, return_all_heads=False):
        input_size = x.shape[-2:]
        
        # Encoder
        x = self.layer0(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        low_level = self.low_level_conv(low_level)
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level], dim=1)
        features = self.decoder_backbone(x)
        
        # Multiple heads
        outputs = [head(features) for head in self.heads]
        
        # Upsample all heads
        outputs = [F.interpolate(out, size=input_size, mode='bilinear', align_corners=True) 
                   for out in outputs]
        
        if return_all_heads:
            return outputs  # List of [B, C, H, W]
        else:
            # Mean voting (ensemble)
            return torch.stack(outputs).mean(dim=0)  # [B, C, H, W]
    
    def get_backbone_params(self):
        """Get backbone parameters"""
        return list(self.layer0.parameters()) + \
               list(self.layer1.parameters()) + \
               list(self.layer2.parameters()) + \
               list(self.layer3.parameters()) + \
               list(self.layer4.parameters())
    
    def get_decoder_params(self):
        """Get decoder parameters"""
        params = list(self.aspp.parameters()) + \
                 list(self.low_level_conv.parameters()) + \
                 list(self.decoder_backbone.parameters())
        for head in self.heads:
            params += list(head.parameters())
        return params


if __name__ == '__main__':
    # Test the models
    print("Testing DeepLabV3Plus_ResNet50...")
    model = DeepLabV3Plus_ResNet50(n_classes=7, output_stride=16, pretrained_backbone=False)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    print("\nTesting MultiHeadDeepLabV3Plus...")
    model = MultiHeadDeepLabV3Plus(n_classes=7, num_heads=3, output_stride=16, pretrained_backbone=False)
    y = model(x, return_all_heads=False)
    print(f"  Input: {x.shape}, Output (ensemble): {y.shape}")
    y_all = model(x, return_all_heads=True)
    print(f"  Output (all heads): {len(y_all)} heads, each {y_all[0].shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
