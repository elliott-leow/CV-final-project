#steps 6-7: em-style training with noisy bump times
#e-step: compute responsibilities w_{k,t}
#m-step: weighted discriminative training

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import config
from model import get_model, count_parameters
from data_generator import load_training_data, organize_by_candidate, get_prior_weights


class EMBumpDataset(Dataset):
    """
    dataset for em training with weighted samples
    handles both positive candidates (with responsibilities) and negatives
    """
    def __init__(self, clips, features, weights, labels):
        self.clips = clips
        self.features = features
        self.weights = weights.astype(np.float32)
        self.labels = labels.astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #normalize and combine
        clip = self.clips[idx].astype(np.float32) / 255.0
        feat = self.features[idx].astype(np.float32) / 255.0
        
        #stack rgb + edges or luma + edges
        if len(feat.shape) == 3:
            feat = feat[..., np.newaxis]
        
        #use rgb + single edge channel for simplicity
        x = np.concatenate([clip, feat[..., :1]], axis=-1)
        
        #transpose to (C, T, H, W)
        x = np.transpose(x, (3, 0, 1, 2))
        
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(self.labels[idx]),
                torch.tensor(self.weights[idx]))


def e_step(model, pos_clips, pos_features, pos_info, device, temperature=1.0,
           old_weights=None, momentum=0.5):
    """
    step 6: e-step - compute soft assignment of true bump frame for each audio event
    
    for each candidate k and each possible bump frame t in T_k:
        w_{k,t} ∝ π_k(t) * p_θ(y=1|C_{k,t})
    normalize so sum over t = 1 for each k
    
    momentum: blend factor for old weights (0=use new, 1=keep old)
    
    returns: updated weights array
    """
    model.eval()
    
    #organize clips by candidate
    candidate_clips = organize_by_candidate(pos_info)
    
    weights = np.zeros(len(pos_info))
    
    with torch.no_grad():
        for k, clip_list in candidate_clips.items():
            #get indices and priors for this candidate
            indices = [c['clip_idx'] for c in clip_list]
            priors = np.array([c['prior_weight'] for c in clip_list])
            
            #compute model predictions for all clips in this candidate
            batch_x = []
            for idx in indices:
                clip = pos_clips[idx].astype(np.float32) / 255.0
                feat = pos_features[idx].astype(np.float32) / 255.0
                
                if len(feat.shape) == 3:
                    feat = feat[..., np.newaxis]
                
                x = np.concatenate([clip, feat[..., :1]], axis=-1)
                x = np.transpose(x, (3, 0, 1, 2))
                batch_x.append(x)
            
            batch_x = torch.tensor(np.array(batch_x), dtype=torch.float32).to(device)
            probs = model.predict_proba(batch_x).cpu().numpy()
            
            #compute responsibilities: w_{k,t} ∝ π_k(t) * p_θ(y=1|C_{k,t})
            log_priors = np.log(priors + 1e-10)
            log_probs = np.log(probs + 1e-10)
            log_weights = (log_priors + log_probs) / temperature
            
            #normalize via softmax
            log_weights = log_weights - log_weights.max()
            responsibilities = np.exp(log_weights)
            responsibilities = responsibilities / (responsibilities.sum() + 1e-10)
            
            #assign to weight array
            for i, idx in enumerate(indices):
                weights[idx] = responsibilities[i]
    
    #blend with old weights if provided (smoothing)
    if old_weights is not None and momentum > 0:
        weights = momentum * old_weights + (1 - momentum) * weights
    
    return weights


def m_step_loss(pred, target, weight, lambda_neg=1.0):
    """
    step 7: compute weighted cross-entropy loss
    
    L_pos = -sum_k sum_t w_{k,t} * log p_θ(y=1|C_{k,t})
    L_neg = -λ * sum_u log(1 - p_θ(y=1|C_u^neg))
    
    pred: model predictions (probabilities)
    target: labels (1 for positive candidates, 0 for negatives)
    weight: responsibilities for positives, 1.0 for negatives
    """
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    
    #positive loss (weighted by responsibility)
    pos_mask = target == 1
    if pos_mask.sum() > 0:
        pos_loss = -weight[pos_mask] * torch.log(pred[pos_mask])
        pos_loss = pos_loss.sum()
    else:
        pos_loss = torch.tensor(0.0, device=pred.device)
    
    #negative loss (weight = 1 for negatives)
    neg_mask = target == 0
    if neg_mask.sum() > 0:
        neg_loss = -lambda_neg * torch.log(1 - pred[neg_mask])
        neg_loss = neg_loss.sum()
    else:
        neg_loss = torch.tensor(0.0, device=pred.device)
    
    total_loss = (pos_loss + neg_loss) / len(pred)
    return total_loss


class FocalLoss(nn.Module):
    """focal loss for hard example mining"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target, weight=None):
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        if weight is not None:
            focal_weight = focal_weight * weight
        
        loss = focal_weight * bce
        return loss.mean()


def train_epoch_em(model, loader, optimizer, device, lambda_neg=1.0, grad_clip=None):
    """train one epoch with weighted loss"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="training", leave=False)
    for batch_x, batch_y, batch_w in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_w = batch_w.to(device)
        
        optimizer.zero_grad()
        
        #get probabilities
        probs = model.predict_proba(batch_x)
        
        #weighted loss
        loss = m_step_loss(probs, batch_y, batch_w, lambda_neg)
        
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    
    return avg_loss, acc


def evaluate_em(model, loader, device, lambda_neg=1.0):
    """evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_w in tqdm(loader, desc="evaluating", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            
            probs = model.predict_proba(batch_x)
            loss = m_step_loss(probs, batch_y, batch_w, lambda_neg)
            
            total_loss += loss.item()
            all_preds.extend(probs.cpu().numpy())
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
    
    return metrics


def train_em(model_type='unet', em_iterations=None, epochs_per_m=None,
             batch_size=None, lr=None):
    """
    steps 6-7: em-style training loop
    
    repeat:
        e-step: recompute w_{k,t} with updated model
        m-step: train with weighted loss for several epochs
    until convergence
    """
    em_iterations = em_iterations or config.EM_ITERATIONS
    epochs_per_m = epochs_per_m or config.EM_EPOCHS_PER_M_STEP
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LEARNING_RATE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    #load training data
    print("loading training data...")
    pos_clips, pos_features, pos_info, neg_clips, neg_features, neg_info = load_training_data()
    print(f"loaded {len(pos_clips)} positive candidates, {len(neg_clips)} negatives")
    
    #initialize weights with priors for positives, 1.0 for negatives
    pos_weights = get_prior_weights(pos_info)
    neg_weights = np.ones(len(neg_clips))
    
    #combine data
    all_clips = np.concatenate([pos_clips, neg_clips], axis=0)
    all_features = np.concatenate([pos_features, neg_features], axis=0)
    all_weights = np.concatenate([pos_weights, neg_weights], axis=0)
    all_labels = np.concatenate([np.ones(len(pos_clips)), np.zeros(len(neg_clips))])
    
    #train/val split
    indices = np.arange(len(all_clips))
    train_idx, val_idx = train_test_split(
        indices, test_size=1-config.TRAIN_VAL_SPLIT,
        stratify=all_labels, random_state=42
    )
    
    print(f"train: {len(train_idx)}, val: {len(val_idx)}")
    
    #create model
    in_channels = 4  #rgb + edge
    model = get_model(model_type, in_channels=in_channels)
    model = model.to(device)
    print(f"model: {model_type}, parameters: {count_parameters(model):,}")
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_val_loss = float('inf')
    history = {'em_iter': [], 'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    print(f"\nstarting EM training: {em_iterations} iterations x {epochs_per_m} epochs/M-step")
    
    for em_iter in range(em_iterations):
        print(f"\n{'='*60}")
        print(f"EM ITERATION {em_iter + 1}/{em_iterations}")
        print(f"{'='*60}")
        
        #e-step: update responsibilities (skip first iteration, use priors)
        if em_iter > 0:
            print("\nE-STEP: computing responsibilities...")
            old_pos_weights = all_weights[:len(pos_clips)].copy()
            new_pos_weights = e_step(
                model, pos_clips, pos_features, pos_info, device,
                temperature=config.RESPONSIBILITY_TEMPERATURE,
                old_weights=old_pos_weights,
                momentum=config.EM_WEIGHT_MOMENTUM
            )
            
            #update weights for positives
            all_weights[:len(pos_clips)] = new_pos_weights
            
            #show weight distribution
            print(f"  weight stats: min={new_pos_weights.min():.4f}, "
                  f"max={new_pos_weights.max():.4f}, "
                  f"mean={new_pos_weights.mean():.4f}")
        
        #m-step: train with current weights
        print(f"\nM-STEP: training for {epochs_per_m} epochs...")
        
        #create datasets with current weights
        train_dataset = EMBumpDataset(
            all_clips[train_idx], all_features[train_idx],
            all_weights[train_idx], all_labels[train_idx]
        )
        val_dataset = EMBumpDataset(
            all_clips[val_idx], all_features[val_idx],
            all_weights[val_idx], all_labels[val_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0, pin_memory=True)
        
        for epoch in range(epochs_per_m):
            train_loss, train_acc = train_epoch_em(
                model, train_loader, optimizer, device,
                lambda_neg=config.LAMBDA_NEG,
                grad_clip=config.GRAD_CLIP
            )
            
            val_metrics = evaluate_em(model, val_loader, device, config.LAMBDA_NEG)
            scheduler.step(val_metrics['loss'])
            
            history['em_iter'].append(em_iter)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])
            
            print(f"  epoch {epoch+1}/{epochs_per_m} | "
                  f"train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_metrics['loss']:.4f} | "
                  f"val_f1: {val_metrics['f1']:.4f}")
            
            #save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_path = os.path.join(config.MODEL_DIR, f'{model_type}_best.pth')
                torch.save({
                    'em_iter': em_iter,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, save_path)
    
    #save final model
    final_path = os.path.join(config.MODEL_DIR, f'{model_type}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, final_path)
    
    final_weights = os.path.join(config.MODEL_DIR, f'{model_type}_final_weights.pth')
    torch.save(model.state_dict(), final_weights)
    
    print(f"\nsaved best model to {config.MODEL_DIR}/{model_type}_best.pth")
    
    return model, history


def plot_training_history(history, save_path=None):
    """plot EM training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    #loss over all epochs
    axes[0].plot(history['train_loss'], label='train', alpha=0.7)
    axes[0].plot(history['val_loss'], label='val', alpha=0.7)
    axes[0].set_xlabel('epoch (across EM iterations)')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].set_title('loss')
    
    #f1 score
    axes[1].plot(history['val_f1'], 'g-', alpha=0.7)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('f1')
    axes[1].set_title('validation f1')
    
    #em iteration markers
    em_iters = np.array(history['em_iter'])
    unique_iters = np.unique(em_iters)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_iters)))
    
    for i, it in enumerate(unique_iters):
        mask = em_iters == it
        idx = np.where(mask)[0]
        axes[2].scatter(idx, np.array(history['val_loss'])[mask], 
                       c=[colors[i]], label=f'EM {it+1}', alpha=0.7)
    
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('val loss')
    axes[2].set_title('loss by EM iteration')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved training plot to {save_path}")
    
    plt.close(fig)
    return fig


#legacy function for backwards compatibility
def train_model(model_type='simple', epochs=None, batch_size=None, lr=None):
    """alias for train_em with single iteration for backward compat"""
    return train_em(
        model_type=model_type,
        em_iterations=1,
        epochs_per_m=epochs or config.NUM_EPOCHS,
        batch_size=batch_size,
        lr=lr
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet_gru',
                       choices=['resnet_gru', 'resnet_cnn', 'resnet_lstm',
                                'efficientnet_gru', 'mobilenet_gru',
                                'lightweight_gru', 'lightweight_cnn'])
    parser.add_argument('--em-iterations', type=int, default=config.EM_ITERATIONS)
    parser.add_argument('--epochs-per-m', type=int, default=config.EM_EPOCHS_PER_M_STEP)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    
    model, history = train_em(
        model_type=args.model,
        em_iterations=args.em_iterations,
        epochs_per_m=args.epochs_per_m,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    plot_path = os.path.join(config.OUTPUT_DIR, f'training_history_{args.model}.png')
    plot_training_history(history, save_path=plot_path)
