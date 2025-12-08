#u-net style temporal cnn for bump detection
#input: (B, T, H, W, C) -> output: (B, 1) bump prediction

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Conv3DBlock(nn.Module):
    """3d convolution block with batchnorm and relu"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """encoder block with two conv layers and pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels)
        self.conv2 = Conv3DBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        features = self.conv2(self.conv1(x))
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """decoder block with upsampling and skip connection"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, 
                                           kernel_size=2, stride=2)
        self.conv1 = Conv3DBlock(in_channels + skip_channels, out_channels)
        self.conv2 = Conv3DBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        #handle size mismatch
        if x.shape != skip.shape:
            diff_t = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [diff_w//2, diff_w - diff_w//2,
                         diff_h//2, diff_h - diff_h//2,
                         diff_t//2, diff_t - diff_t//2])
        
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class TemporalBumpDetector(nn.Module):
    """
    u-net style architecture for temporal bump detection
    input: (B, C, T, H, W) - batch, channels, time, height, width
    output: (B, 1) - bump probability
    """
    def __init__(self, in_channels=3, base_filters=32):
        super().__init__()
        
        #encoder path
        self.enc1 = EncoderBlock(in_channels, base_filters)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)
        
        #bottleneck
        self.bottleneck = nn.Sequential(
            Conv3DBlock(base_filters * 8, base_filters * 16),
            Conv3DBlock(base_filters * 16, base_filters * 16)
        )
        
        #decoder path with skip connections
        self.dec4 = DecoderBlock(base_filters * 16, base_filters * 8, base_filters * 8)
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2)
        self.dec1 = DecoderBlock(base_filters * 2, base_filters, base_filters)
        
        #global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters, base_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_filters, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        #encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        #bottleneck
        x = self.bottleneck(x)
        
        #decoder with skip connections
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        #classification
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x.squeeze(-1)


class SimplerTemporalCNN(nn.Module):
    """
    simpler 3d cnn without u-net decoder
    faster training for initial experiments
    """
    def __init__(self, in_channels=3, base_filters=32):
        super().__init__()
        
        self.features = nn.Sequential(
            #block 1
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            #block 2
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            #block 3
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            #block 4
            nn.Conv3d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_filters * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)


class TemporalAttentionCNN(nn.Module):
    """
    cnn with temporal attention mechanism
    learns to focus on relevant temporal regions
    """
    def __init__(self, in_channels=3, base_filters=32):
        super().__init__()
        
        #spatial feature extraction (2d conv per frame)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        #temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(base_filters * 4 * 16, base_filters),
            nn.ReLU(inplace=True),
            nn.Linear(base_filters, 1),
            nn.Softmax(dim=1)
        )
        
        #temporal fusion
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(base_filters * 4 * 16, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(base_filters * 4, base_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_filters, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        #x shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        #process each frame independently
        x = x.permute(0, 2, 1, 3, 4)  #(B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        
        spatial_features = self.spatial_encoder(x)  #(B*T, F, 4, 4)
        spatial_features = spatial_features.view(B, T, -1)  #(B, T, F*16)
        
        #compute temporal attention weights
        attn_weights = self.temporal_attention(spatial_features)  #(B, T, 1)
        
        #weighted sum of temporal features
        weighted = spatial_features * attn_weights
        
        #temporal convolution
        temporal_features = spatial_features.permute(0, 2, 1)  #(B, F*16, T)
        temporal_out = self.temporal_conv(temporal_features)
        temporal_out = temporal_out.mean(dim=2)  #global temporal pooling
        
        #classify
        output = self.classifier(temporal_out)
        return output.squeeze(-1)


def get_model(model_type='unet', in_channels=3, base_filters=32):
    """factory function to get model by type"""
    if model_type == 'unet':
        return TemporalBumpDetector(in_channels, base_filters)
    elif model_type == 'simple':
        return SimplerTemporalCNN(in_channels, base_filters)
    elif model_type == 'attention':
        return TemporalAttentionCNN(in_channels, base_filters)
    else:
        raise ValueError(f"unknown model type: {model_type}")


def count_parameters(model):
    """count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    #test models with sample input
    batch_size = 2
    channels = 4  #rgb + edge
    time_steps = 15
    height = 240
    width = 320
    
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    print("testing models...")
    
    for model_type in ['unet', 'simple', 'attention']:
        print(f"\n{model_type.upper()} model:")
        model = get_model(model_type, in_channels=channels)
        print(f"  parameters: {count_parameters(model):,}")
        
        #forward pass
        with torch.no_grad():
            output = model(x)
        print(f"  input shape: {x.shape}")
        print(f"  output shape: {output.shape}")
        print(f"  output values: {output}")

