#training data generation from processed video and ground truth
#creates clips for bump detection training

import numpy as np
import os
from tqdm import tqdm
import cv2

import config


def create_training_clips(frames, edge_frames, bump_frames, 
                          frames_before=None, clip_length=None,
                          min_frames_to_next=None):
    """
    create training clips from video frames and bump annotations
    
    positive samples: clips ending at bump frame
    negative samples: clips with no nearby bumps
    """
    frames_before = frames_before or config.FRAMES_BEFORE_BUMP
    clip_length = clip_length or config.CLIP_LENGTH
    min_frames_to_next = min_frames_to_next or config.MIN_FRAMES_TO_NEXT_BUMP
    
    bump_set = set(bump_frames)
    total_frames = len(frames)
    
    clips = []
    edge_clips = []
    labels = []
    clip_info = []
    
    #create positive samples
    print("creating positive samples...")
    valid_bumps = 0
    for bump_frame in tqdm(bump_frames, desc="positive clips"):
        start_frame = bump_frame - frames_before
        end_frame = start_frame + clip_length
        
        if start_frame < 0 or end_frame > total_frames:
            continue
        
        clip = frames[start_frame:end_frame]
        edge_clip = edge_frames[start_frame:end_frame]
        
        clips.append(clip)
        edge_clips.append(edge_clip)
        labels.append(1)
        clip_info.append({
            'start': start_frame,
            'end': end_frame,
            'bump_frame': bump_frame,
            'type': 'positive'
        })
        valid_bumps += 1
    
    print(f"  created {valid_bumps} positive clips")
    
    #create negative samples
    print("creating negative samples...")
    
    #precompute distance to nearest future bump for each frame
    bump_frames_sorted = np.sort(bump_frames)
    
    negative_starts = []
    for frame_idx in tqdm(range(0, total_frames - clip_length, 5), desc="finding negatives"):  #sample every 5 frames
        end_frame = frame_idx + clip_length
        
        #check if any bump is within clip
        has_bump = np.any((bump_frames_sorted >= frame_idx) & (bump_frames_sorted < end_frame))
        if has_bump:
            continue
        
        #check distance to next bump after clip
        future_bumps = bump_frames_sorted[bump_frames_sorted >= end_frame]
        if len(future_bumps) > 0:
            dist_to_next = future_bumps[0] - end_frame
            if dist_to_next >= min_frames_to_next:
                negative_starts.append(frame_idx)
        else:
            negative_starts.append(frame_idx)
    
    #sample negatives to balance
    num_positives = len([l for l in labels if l == 1])
    num_negatives_needed = min(num_positives * 2, len(negative_starts))
    
    if len(negative_starts) > 0:
        step = max(1, len(negative_starts) // num_negatives_needed)
        sampled_starts = negative_starts[::step][:num_negatives_needed]
        
        for start_frame in tqdm(sampled_starts, desc="negative clips"):
            end_frame = start_frame + clip_length
            
            clip = frames[start_frame:end_frame]
            edge_clip = edge_frames[start_frame:end_frame]
            
            clips.append(clip)
            edge_clips.append(edge_clip)
            labels.append(0)
            clip_info.append({
                'start': start_frame,
                'end': end_frame,
                'bump_frame': None,
                'type': 'negative'
            })
    
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    print(f"created {num_pos} positive and {num_neg} negative clips")
    
    return np.array(clips), np.array(edge_clips), np.array(labels), clip_info


def save_training_data(clips, edge_clips, labels, clip_info, save_dir=None):
    """save training data to disk"""
    save_dir = save_dir or config.TRAINING_DATA_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print("saving training data...")
    np.save(os.path.join(save_dir, 'clips.npy'), clips)
    np.save(os.path.join(save_dir, 'edge_clips.npy'), edge_clips)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    np.save(os.path.join(save_dir, 'clip_info.npy'), clip_info)
    
    print(f"saved to {save_dir}")
    print(f"  clips: {clips.shape}")
    print(f"  edge_clips: {edge_clips.shape}")
    print(f"  labels: {labels.shape}")


def load_training_data(load_dir=None):
    """load training data from disk"""
    load_dir = load_dir or config.TRAINING_DATA_DIR
    
    print("loading training data...")
    clips = np.load(os.path.join(load_dir, 'clips.npy'))
    edge_clips = np.load(os.path.join(load_dir, 'edge_clips.npy'))
    labels = np.load(os.path.join(load_dir, 'labels.npy'))
    clip_info = np.load(os.path.join(load_dir, 'clip_info.npy'), allow_pickle=True)
    
    print(f"loaded {len(clips)} clips")
    return clips, edge_clips, labels, clip_info


def create_combined_features(clips, edge_clips):
    """combine RGB clips with edge features"""
    rgb_normalized = clips.astype(np.float32) / 255.0
    edges_normalized = edge_clips.astype(np.float32) / 255.0
    edges_normalized = np.expand_dims(edges_normalized, axis=-1)
    combined = np.concatenate([rgb_normalized, edges_normalized], axis=-1)
    return combined


def visualize_training_samples(clips, labels, num_samples=4, save_path=None):
    """visualize sample training clips"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))
    
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    sample_indices = []
    n_pos = min(num_samples // 2, len(pos_indices))
    n_neg = min(num_samples // 2, len(neg_indices))
    
    if n_pos > 0:
        sample_indices.extend(np.random.choice(pos_indices, n_pos, replace=False))
    if n_neg > 0:
        sample_indices.extend(np.random.choice(neg_indices, n_neg, replace=False))
    
    for row, idx in enumerate(sample_indices[:num_samples]):
        clip = clips[idx]
        label = labels[idx]
        
        frame_indices = np.linspace(0, len(clip) - 1, 5).astype(int)
        
        for col, frame_idx in enumerate(frame_indices):
            axes[row, col].imshow(clip[frame_idx])
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(f"{'BUMP' if label else 'NO BUMP'}", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved visualization to {save_path}")
    
    plt.close(fig)
    return fig


if __name__ == "__main__":
    from video_scaler import scale_video, load_scaled_video
    from edge_processor import process_edges
    from audio_ground_truth import generate_ground_truth
    
    video_path = os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4")
    
    if os.path.exists(video_path):
        scaled_path = os.path.join(config.OUTPUT_DIR, "scaled_video.mp4")
        gt_path = os.path.join(config.OUTPUT_DIR, "ground_truth.npy")
        
        if not os.path.exists(scaled_path):
            frames, _ = scale_video(video_path, scaled_path)
        else:
            frames = load_scaled_video(scaled_path)
        
        if not os.path.exists(gt_path):
            gt = generate_ground_truth(video_path, save_path=gt_path)
        else:
            gt = np.load(gt_path, allow_pickle=True).item()
        
        edge_results = process_edges(frames)
        edge_frames = edge_results['filtered']
        
        original_fps = gt['video_fps']
        bump_frames_original = gt['bump_frames']
        fps_ratio = config.TARGET_FPS / original_fps
        bump_frames_scaled = (bump_frames_original * fps_ratio).astype(int)
        
        clips, edge_clips, labels, clip_info = create_training_clips(
            frames, edge_frames, bump_frames_scaled
        )
        
        save_training_data(clips, edge_clips, labels, clip_info)
        
        viz_path = os.path.join(config.OUTPUT_DIR, "training_samples.png")
        visualize_training_samples(clips, labels, save_path=viz_path)
    else:
        print(f"video not found: {video_path}")
