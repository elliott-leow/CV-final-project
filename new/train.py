#training script for bump detection model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import config
from model import get_model, count_parameters
from data_generator import load_training_data


class FocalLoss(nn.Module):
    """
    Focal Loss - focuses learning on hard examples
    Reduces loss for well-classified examples, emphasizing hard ones
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # balance factor for positive/negative
        self.gamma = gamma  # focusing parameter (higher = more focus on hard examples)
    
    def forward(self, pred, target):
        # pred should be probabilities (after sigmoid)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # pt is the probability of the correct class
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # alpha weighting
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # focal term: (1 - pt)^gamma reduces loss for easy examples
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        loss = focal_weight * bce
        return loss.mean()


class SpikeyLoss(nn.Module):
    """
    Custom loss that encourages spiky predictions:
    1. Focal loss for hard example mining
    2. Margin loss to push predictions away from 0.5
    3. Class-specific confidence targets (high for bumps, low for non-bumps)
    """
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, 
                 margin_weight=0.3, pos_target=0.9, neg_target=0.1):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.margin_weight = margin_weight
        self.pos_target = pos_target  # target probability for positive samples
        self.neg_target = neg_target  # target probability for negative samples
    
    def forward(self, pred, target):
        # Base focal loss
        focal_loss = self.focal(pred, target)
        
        # Margin loss: penalize predictions that aren't confident enough
        # For positive samples: push towards pos_target (e.g., 0.9)
        # For negative samples: push towards neg_target (e.g., 0.1)
        soft_targets = torch.where(target == 1, 
                                   torch.full_like(pred, self.pos_target),
                                   torch.full_like(pred, self.neg_target))
        margin_loss = F.mse_loss(pred, soft_targets)
        
        total_loss = focal_loss + self.margin_weight * margin_loss
        return total_loss


class BumpDataset(Dataset):
    """pytorch dataset for bump detection clips - lazy loading"""
    def __init__(self, clips, edge_clips, labels, augment=False):
        self.clips = clips
        self.edge_clips = edge_clips
        self.labels = labels.astype(np.float32)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #load and normalize on-the-fly to save memory
        clip = self.clips[idx].astype(np.float32) / 255.0
        edge = self.edge_clips[idx].astype(np.float32) / 255.0
        edge = np.expand_dims(edge, axis=-1)
        
        #combine rgb + edge
        x = np.concatenate([clip, edge], axis=-1)
        
        #transpose to (C, T, H, W) for pytorch
        x = np.transpose(x, (3, 0, 1, 2))
        
        if self.augment:
            x = self._augment(x)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.labels[idx])
    
    def _augment(self, x):
        """apply data augmentation"""
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=3).copy()
        
        if np.random.rand() > 0.5:
            factor = 0.8 + np.random.rand() * 0.4
            x[:3] = np.clip(x[:3] * factor, 0, 1)
        
        return x


class EarlyStopping:
    """early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, loader, criterion, optimizer, device):
    """train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="training", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    """evaluate model on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="evaluating", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    labels_array = np.array(all_labels)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(labels_array, preds_binary),
        'precision': precision_score(labels_array, preds_binary, zero_division=0),
        'recall': recall_score(labels_array, preds_binary, zero_division=0),
        'f1': f1_score(labels_array, preds_binary, zero_division=0),
    }
    
    if len(np.unique(labels_array)) > 1:
        metrics['auc'] = roc_auc_score(labels_array, all_preds)
    else:
        metrics['auc'] = 0.0
    
    return metrics


def train_model(model_type='simple', epochs=None, batch_size=None, lr=None):
    """main training function"""
    epochs = epochs or config.NUM_EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LEARNING_RATE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    print("loading training data...")
    clips, edge_clips, labels, _ = load_training_data()
    print(f"loaded {len(clips)} clips, {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    #train/val split
    indices = np.arange(len(clips))
    train_idx, val_idx = train_test_split(
        indices, test_size=1-config.TRAIN_VAL_SPLIT, 
        stratify=labels, random_state=42
    )
    
    print(f"train samples: {len(train_idx)}, val samples: {len(val_idx)}")
    
    train_dataset = BumpDataset(clips[train_idx], edge_clips[train_idx], 
                                labels[train_idx], augment=True)
    val_dataset = BumpDataset(clips[val_idx], edge_clips[val_idx], 
                              labels[val_idx], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    #create model
    in_channels = 4  #rgb + edge
    model = get_model(model_type, in_channels=in_channels)
    model = model.to(device)
    print(f"model: {model_type}, parameters: {count_parameters(model):,}")
    
    #loss and optimizer - use SpikeyLoss for better peak detection
    criterion = SpikeyLoss(
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA,
        margin_weight=config.MARGIN_WEIGHT,
        pos_target=config.POS_TARGET,
        neg_target=config.NEG_TARGET
    )
    print(f"using SpikeyLoss (focal_gamma={config.FOCAL_GAMMA}, margin={config.MARGIN_WEIGHT})")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    #training loop
    early_stopping = EarlyStopping(patience=10)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 
               'val_f1': [], 'val_auc': []}
    best_val_loss = float('inf')
    
    print(f"\nstarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['loss'])
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            #save full checkpoint (for resuming training)
            save_path = os.path.join(config.MODEL_DIR, f'{model_type}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, save_path)
            
            #save weights only (for inference)
            weights_path = os.path.join(config.MODEL_DIR, f'{model_type}_weights.pth')
            torch.save(model.state_dict(), weights_path)
        
        print(f"epoch {epoch+1}/{epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} | "
              f"val_acc: {val_metrics['accuracy']:.4f} | "
              f"val_f1: {val_metrics['f1']:.4f} | "
              f"val_auc: {val_metrics['auc']:.4f}")
        
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f"early stopping at epoch {epoch+1}")
            break
    
    #save final model (checkpoint)
    final_path = os.path.join(config.MODEL_DIR, f'{model_type}_final.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, final_path)
    
    #save final weights only
    final_weights = os.path.join(config.MODEL_DIR, f'{model_type}_final_weights.pth')
    torch.save(model.state_dict(), final_weights)
    
    print(f"saved best model to {config.MODEL_DIR}/{model_type}_best.pth")
    print(f"saved weights only to {config.MODEL_DIR}/{model_type}_weights.pth")
    
    return model, history


def plot_training_history(history, save_path=None):
    """plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='train')
    axes[0, 0].plot(history['val_loss'], label='val')
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('loss')
    
    axes[0, 1].plot(history['val_acc'], label='val accuracy')
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('accuracy')
    axes[0, 1].set_title('validation accuracy')
    
    axes[1, 0].plot(history['val_f1'], label='val f1')
    axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylabel('f1 score')
    axes[1, 0].set_title('validation f1 score')
    
    axes[1, 1].plot(history['val_auc'], label='val auc')
    axes[1, 1].set_xlabel('epoch')
    axes[1, 1].set_ylabel('auc')
    axes[1, 1].set_title('validation auc')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved training plot to {save_path}")
    
    plt.close(fig)
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple', 
                       choices=['unet', 'simple', 'attention'])
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    
    model, history = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    plot_path = os.path.join(config.OUTPUT_DIR, f'training_history_{args.model}.png')
    plot_training_history(history, save_path=plot_path)
