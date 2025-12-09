#step 5: pretrained 2d cnn encoder + temporal head for bump detection
#input: (B, C, T, H, W) -> output: (B,) bump probability

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import config


class PretrainedEncoder(nn.Module):
    """
    pretrained 2d cnn encoder for frame-level feature extraction
    uses resnet18/34/50 or efficientnet as backbone
    """
    def __init__(self, backbone='resnet18', pretrained=True, freeze_early=True):
        super().__init__()
        
        if backbone == 'resnet18':
            base = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512
            #remove fc layer
            self.encoder = nn.Sequential(*list(base.children())[:-2])
        
        elif backbone == 'resnet34':
            base = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512
            self.encoder = nn.Sequential(*list(base.children())[:-2])
        
        elif backbone == 'resnet50':
            base = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 2048
            self.encoder = nn.Sequential(*list(base.children())[:-2])
        
        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 1280
            self.encoder = base.features
        
        elif backbone == 'mobilenet_v3':
            base = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 576
            self.encoder = base.features
        
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        #freeze early layers for transfer learning
        if freeze_early and pretrained:
            self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """freeze first few layers of pretrained model"""
        for i, child in enumerate(self.encoder.children()):
            if i < 4:  #freeze first 4 blocks
                for param in child.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        #x: (B, C, H, W)
        features = self.encoder(x)  #(B, F, h, w)
        features = self.pool(features)  #(B, F, 1, 1)
        features = features.flatten(1)  #(B, F)
        return features


class TemporalCNNHead(nn.Module):
    """
    1d temporal cnn head for aggregating frame features
    processes sequence of frame features -> single clip prediction
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x):
        #x: (B, T, F) - batch of frame feature sequences
        x = x.permute(0, 2, 1)  #(B, F, T)
        x = self.conv_layers(x)  #(B, H, T)
        x = self.pool(x).squeeze(-1)  #(B, H)
        logits = self.classifier(x)  #(B, 1)
        return logits.squeeze(-1)


class TemporalGRUHead(nn.Module):
    """
    gru-based temporal head for aggregating frame features
    bidirectional gru with attention pooling
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3,
                 bidirectional=True):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        #attention for temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        #x: (B, T, F) - batch of frame feature sequences
        gru_out, _ = self.gru(x)  #(B, T, H*2)
        
        #attention pooling
        attn_scores = self.attention(gru_out)  #(B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  #(B, T, 1)
        
        #weighted sum
        context = (gru_out * attn_weights).sum(dim=1)  #(B, H*2)
        
        logits = self.classifier(context)  #(B, 1)
        return logits.squeeze(-1)


class TemporalLSTMHead(nn.Module):
    """
    lstm-based temporal head with last hidden state
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        #x: (B, T, F)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        #concat last forward and backward hidden states
        h_forward = h_n[-2]  #(B, H)
        h_backward = h_n[-1]  #(B, H)
        h_final = torch.cat([h_forward, h_backward], dim=1)  #(B, H*2)
        
        logits = self.classifier(h_final)
        return logits.squeeze(-1)


class FrameEncoderTemporalModel(nn.Module):
    """
    pretrained 2d cnn frame encoder + temporal head
    
    architecture:
        1. encode each frame with pretrained cnn (resnet/efficientnet)
        2. aggregate frame features with temporal head (1d cnn or gru)
        3. output single probability per clip
    
    input: X ∈ R^(B, C, T, H, W) where T=15, H=240, W=320
    output: p_θ(y=1|C) ∈ [0,1]
    """
    def __init__(self, in_channels=4, backbone='resnet18', temporal_head='gru',
                 hidden_dim=256, pretrained=True, freeze_early=True, dropout=0.3):
        super().__init__()
        
        self.in_channels = in_channels
        
        #input projection if channels != 3
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_proj = None
        
        #pretrained frame encoder
        self.frame_encoder = PretrainedEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze_early=freeze_early
        )
        
        feature_dim = self.frame_encoder.feature_dim
        
        #temporal head
        if temporal_head == 'cnn':
            self.temporal_head = TemporalCNNHead(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif temporal_head == 'gru':
            self.temporal_head = TemporalGRUHead(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif temporal_head == 'lstm':
            self.temporal_head = TemporalLSTMHead(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"unknown temporal head: {temporal_head}")
        
        self.backbone_name = backbone
        self.temporal_name = temporal_head
    
    def forward(self, x):
        #x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        #reshape to process frames independently
        x = x.permute(0, 2, 1, 3, 4)  #(B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)  #(B*T, C, H, W)
        
        #project to 3 channels if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        #encode each frame
        frame_features = self.frame_encoder(x)  #(B*T, F)
        
        #reshape back to sequence
        frame_features = frame_features.view(B, T, -1)  #(B, T, F)
        
        #temporal aggregation
        logits = self.temporal_head(frame_features)  #(B,)
        
        return logits
    
    def predict_proba(self, x):
        """get probability output"""
        logits = self.forward(x)
        return torch.sigmoid(logits)


class LightweightEncoder(nn.Module):
    """
    lightweight custom encoder for faster training
    useful when pretrained models are too heavy
    """
    def __init__(self, in_channels=4, base_filters=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            #block 1
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            #block 2
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            #block 3
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            #block 4
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.feature_dim = base_filters * 8
    
    def forward(self, x):
        features = self.encoder(x)
        return features.flatten(1)


class LightweightTemporalModel(nn.Module):
    """
    lightweight model with custom encoder + temporal head
    faster training, smaller memory footprint
    """
    def __init__(self, in_channels=4, base_filters=64, temporal_head='gru',
                 hidden_dim=128, dropout=0.3):
        super().__init__()
        
        self.frame_encoder = LightweightEncoder(in_channels, base_filters)
        feature_dim = self.frame_encoder.feature_dim
        
        if temporal_head == 'cnn':
            self.temporal_head = TemporalCNNHead(feature_dim, hidden_dim, dropout=dropout)
        elif temporal_head == 'gru':
            self.temporal_head = TemporalGRUHead(feature_dim, hidden_dim, dropout=dropout)
        elif temporal_head == 'lstm':
            self.temporal_head = TemporalLSTMHead(feature_dim, hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"unknown temporal head: {temporal_head}")
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        #process frames
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frame_features = self.frame_encoder(x).view(B, T, -1)
        
        #temporal
        logits = self.temporal_head(frame_features)
        return logits
    
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


def get_model(model_type='resnet_gru', in_channels=4, base_filters=32, dropout=0.1):
    """
    factory function to get model by type
    
    model types:
        - resnet_gru: resnet18 encoder + gru temporal head (recommended)
        - resnet_cnn: resnet18 encoder + 1d cnn temporal head
        - efficientnet_gru: efficientnet_b0 encoder + gru head
        - mobilenet_gru: mobilenet_v3 encoder + gru head (lightweight)
        - lightweight_gru: custom lightweight encoder + gru head
        - lightweight_cnn: custom lightweight encoder + cnn head
    
    legacy types (backwards compat):
        - unet, simple, attention: map to resnet_gru
    """
    #legacy mappings
    if model_type in ['unet', 'simple', 'attention']:
        model_type = 'resnet_gru'
    
    if model_type == 'resnet_gru':
        return FrameEncoderTemporalModel(
            in_channels=in_channels,
            backbone='resnet18',
            temporal_head='gru',
            hidden_dim=256,
            pretrained=True,
            dropout=dropout
        )
    
    elif model_type == 'resnet_cnn':
        return FrameEncoderTemporalModel(
            in_channels=in_channels,
            backbone='resnet18',
            temporal_head='cnn',
            hidden_dim=256,
            pretrained=True,
            dropout=dropout
        )
    
    elif model_type == 'resnet_lstm':
        return FrameEncoderTemporalModel(
            in_channels=in_channels,
            backbone='resnet18',
            temporal_head='lstm',
            hidden_dim=256,
            pretrained=True,
            dropout=dropout
        )
    
    elif model_type == 'efficientnet_gru':
        return FrameEncoderTemporalModel(
            in_channels=in_channels,
            backbone='efficientnet_b0',
            temporal_head='gru',
            hidden_dim=256,
            pretrained=True,
            dropout=dropout
        )
    
    elif model_type == 'mobilenet_gru':
        return FrameEncoderTemporalModel(
            in_channels=in_channels,
            backbone='mobilenet_v3',
            temporal_head='gru',
            hidden_dim=128,
            pretrained=True,
            dropout=dropout
        )
    
    elif model_type == 'lightweight_gru':
        return LightweightTemporalModel(
            in_channels=in_channels,
            base_filters=64,
            temporal_head='gru',
            hidden_dim=128,
            dropout=dropout
        )
    
    elif model_type == 'lightweight_cnn':
        return LightweightTemporalModel(
            in_channels=in_channels,
            base_filters=64,
            temporal_head='cnn',
            hidden_dim=128,
            dropout=dropout
        )
    
    else:
        raise ValueError(f"unknown model type: {model_type}")


def count_parameters(model):
    """count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """count all parameters (including frozen)"""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    #test models with sample input
    batch_size = 2
    channels = 4  #rgb + edge
    time_steps = 15
    height = 240
    width = 320
    
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    print("testing models...")
    print(f"input shape: {x.shape}")
    print()
    
    model_types = [
        'resnet_gru', 'resnet_cnn', 'resnet_lstm',
        'mobilenet_gru', 'lightweight_gru', 'lightweight_cnn'
    ]
    
    for model_type in model_types:
        print(f"{model_type}:")
        model = get_model(model_type, in_channels=channels)
        trainable = count_parameters(model)
        total = count_all_parameters(model)
        print(f"  trainable params: {trainable:,}")
        print(f"  total params: {total:,}")
        
        #forward pass
        with torch.no_grad():
            logits = model(x)
            probs = model.predict_proba(x)
        print(f"  output shape: {logits.shape}")
        print(f"  probs: {probs.tolist()}")
        print()
