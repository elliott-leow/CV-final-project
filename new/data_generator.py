#step 3: clip extraction with noisy labels
#extracts positive candidate clips and negative clips

import numpy as np
import os
from tqdm import tqdm
import cv2

import config


def extract_candidate_clips(frames, feature_frames, candidates,
                            horizon=None, clip_length=None):
    """
    step 3a: extract candidate positive clips for each bump candidate
    
    for each candidate k and each possible true bump frame t in T_k:
        - clip start s = t - H (horizon frames before bump)
        - extract clip C_{k,t} = frames[s:s+T_clip]
    
    args:
        frames: video frames (T, H, W, C)
        feature_frames: processed feature frames (T, H, W, C_feat)
        candidates: list of candidate dicts from audio detection
        horizon: H - early warning horizon (frames before bump)
        clip_length: T_clip - length of each clip
    
    returns:
        clips: list of clip arrays
        feature_clips: list of feature clip arrays
        clip_info: list of dicts with metadata (k, t, prior weight, etc)
    """
    horizon = horizon or config.EARLY_WARNING_HORIZON
    clip_length = clip_length or config.CLIP_LENGTH
    total_frames = len(frames)
    
    clips = []
    feature_clips = []
    clip_info = []
    
    print(f"extracting candidate positive clips...")
    print(f"  horizon H={horizon}, clip length={clip_length}")
    
    for k, cand in enumerate(tqdm(candidates, desc="candidate clips")):
        window = cand['window']
        prior = cand['prior']
        
        for i, t in enumerate(window):
            #clip start is t - H (predict bump at t, starting H frames before)
            s = t - horizon
            
            #check bounds
            if s < 0 or s + clip_length > total_frames:
                continue
            
            clip = frames[s:s+clip_length]
            feat_clip = feature_frames[s:s+clip_length]
            
            clips.append(clip)
            feature_clips.append(feat_clip)
            clip_info.append({
                'candidate_idx': k,
                'bump_frame': t,
                'clip_start': s,
                'prior_weight': prior[i],
                'audio_frame': cand['audio_frame'],
                'type': 'positive_candidate'
            })
    
    print(f"  extracted {len(clips)} candidate positive clips")
    return clips, feature_clips, clip_info


def get_candidate_exclusion_zones(candidates, clip_length, min_distance=None):
    """get frame ranges to exclude for negative sampling"""
    min_distance = min_distance or config.MIN_NEGATIVE_DISTANCE
    
    exclusion_ranges = []
    for cand in candidates:
        window = cand['window']
        t_min = window.min()
        t_max = window.max()
        
        #exclude zone: any clip that could contain frames near candidate
        #clip at start s covers frames [s, s+clip_length-1]
        #for safety, exclude starts that put any frame within min_distance of window
        exclude_start = t_min - clip_length - min_distance + 1
        exclude_end = t_max + min_distance
        
        exclusion_ranges.append((exclude_start, exclude_end))
    
    return exclusion_ranges


def is_valid_negative_start(start, exclusion_ranges, clip_length, total_frames):
    """check if a clip start is valid for negative sampling"""
    if start < 0 or start + clip_length > total_frames:
        return False
    
    clip_end = start + clip_length - 1
    
    for ex_start, ex_end in exclusion_ranges:
        #check if clip overlaps with exclusion zone
        if not (clip_end < ex_start or start > ex_end):
            return False
    
    return True


def extract_negative_clips(frames, feature_frames, candidates,
                           clip_length=None, num_negatives=None, 
                           min_distance=None, sample_step=5):
    """
    step 3b: extract negative clips far from any bump candidate
    
    negative clips are sampled from regions at least min_distance frames
    away from any candidate window T_k
    
    args:
        frames: video frames
        feature_frames: processed feature frames
        candidates: list of candidate dicts
        clip_length: length of each clip
        num_negatives: target number of negative clips
        min_distance: minimum frames from any candidate
        sample_step: step for scanning valid positions
    
    returns:
        clips, feature_clips, clip_info (with label=0, weight=1)
    """
    clip_length = clip_length or config.CLIP_LENGTH
    min_distance = min_distance or config.MIN_NEGATIVE_DISTANCE
    total_frames = len(frames)
    
    #get exclusion zones
    exclusion_ranges = get_candidate_exclusion_zones(candidates, clip_length, min_distance)
    
    #find valid negative start positions
    print(f"finding negative clip positions (min {min_distance} frames from candidates)...")
    valid_starts = []
    for s in range(0, total_frames - clip_length, sample_step):
        if is_valid_negative_start(s, exclusion_ranges, clip_length, total_frames):
            valid_starts.append(s)
    
    print(f"  found {len(valid_starts)} valid negative positions")
    
    if num_negatives is None:
        #default: 2x positives (count all candidate clips)
        num_candidates = sum(len(c['window']) for c in candidates)
        num_negatives = min(num_candidates * config.NEGATIVES_PER_POSITIVE, len(valid_starts))
    
    #sample evenly from valid positions
    if len(valid_starts) > num_negatives:
        step = len(valid_starts) // num_negatives
        sampled_starts = valid_starts[::step][:num_negatives]
    else:
        sampled_starts = valid_starts
    
    clips = []
    feature_clips = []
    clip_info = []
    
    for s in tqdm(sampled_starts, desc="negative clips"):
        clip = frames[s:s+clip_length]
        feat_clip = feature_frames[s:s+clip_length]
        
        clips.append(clip)
        feature_clips.append(feat_clip)
        clip_info.append({
            'clip_start': s,
            'weight': 1.0,
            'type': 'negative'
        })
    
    print(f"  extracted {len(clips)} negative clips")
    return clips, feature_clips, clip_info


def create_training_data(frames, feature_frames, candidates,
                         horizon=None, clip_length=None, 
                         min_distance=None, neg_ratio=None):
    """
    step 3: create complete training dataset with candidate positive and negative clips
    
    returns:
        positive_clips: candidate positive clips (noisy labels)
        positive_features: feature clips for positives
        positive_info: metadata including prior weights
        negative_clips: definite negative clips
        negative_features: feature clips for negatives
        negative_info: metadata
    """
    horizon = horizon or config.EARLY_WARNING_HORIZON
    clip_length = clip_length or config.CLIP_LENGTH
    min_distance = min_distance or config.MIN_NEGATIVE_DISTANCE
    neg_ratio = neg_ratio or config.NEGATIVES_PER_POSITIVE
    
    print("\nstep 3: extracting training clips...")
    print(f"  horizon: {horizon} frames")
    print(f"  clip length: {clip_length} frames")
    
    #extract candidate positives
    pos_clips, pos_features, pos_info = extract_candidate_clips(
        frames, feature_frames, candidates, horizon, clip_length
    )
    
    #extract negatives
    num_negatives = int(len(pos_clips) * neg_ratio / len(candidates)) if candidates else 0
    neg_clips, neg_features, neg_info = extract_negative_clips(
        frames, feature_frames, candidates, clip_length, 
        num_negatives, min_distance
    )
    
    print(f"\ntotal clips: {len(pos_clips)} candidates, {len(neg_clips)} negatives")
    
    return (np.array(pos_clips), np.array(pos_features), pos_info,
            np.array(neg_clips), np.array(neg_features), neg_info)


def save_training_data(pos_clips, pos_features, pos_info,
                       neg_clips, neg_features, neg_info,
                       save_dir=None):
    """save training data to disk"""
    save_dir = save_dir or config.TRAINING_DATA_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print("saving training data...")
    np.save(os.path.join(save_dir, 'pos_clips.npy'), pos_clips)
    np.save(os.path.join(save_dir, 'pos_features.npy'), pos_features)
    np.save(os.path.join(save_dir, 'pos_info.npy'), pos_info, allow_pickle=True)
    np.save(os.path.join(save_dir, 'neg_clips.npy'), neg_clips)
    np.save(os.path.join(save_dir, 'neg_features.npy'), neg_features)
    np.save(os.path.join(save_dir, 'neg_info.npy'), neg_info, allow_pickle=True)
    
    print(f"saved to {save_dir}")
    print(f"  positive clips: {pos_clips.shape}")
    print(f"  negative clips: {neg_clips.shape}")


def load_training_data(load_dir=None):
    """load training data from disk"""
    load_dir = load_dir or config.TRAINING_DATA_DIR
    
    print("loading training data...")
    pos_clips = np.load(os.path.join(load_dir, 'pos_clips.npy'))
    pos_features = np.load(os.path.join(load_dir, 'pos_features.npy'))
    pos_info = np.load(os.path.join(load_dir, 'pos_info.npy'), allow_pickle=True)
    neg_clips = np.load(os.path.join(load_dir, 'neg_clips.npy'))
    neg_features = np.load(os.path.join(load_dir, 'neg_features.npy'))
    neg_info = np.load(os.path.join(load_dir, 'neg_info.npy'), allow_pickle=True)
    
    print(f"loaded {len(pos_clips)} positive candidates, {len(neg_clips)} negatives")
    return pos_clips, pos_features, pos_info, neg_clips, neg_features, neg_info


def organize_by_candidate(pos_info):
    """organize positive clip indices by candidate for EM training"""
    candidate_clips = {}
    for i, info in enumerate(pos_info):
        k = info['candidate_idx']
        if k not in candidate_clips:
            candidate_clips[k] = []
        candidate_clips[k].append({
            'clip_idx': i,
            'bump_frame': info['bump_frame'],
            'prior_weight': info['prior_weight']
        })
    return candidate_clips


def get_prior_weights(pos_info):
    """extract prior weights as array for training"""
    return np.array([info['prior_weight'] for info in pos_info])


def create_combined_features(clips, feature_clips):
    """combine RGB clips with processed features (edges)"""
    rgb_normalized = clips.astype(np.float32) / 255.0
    feat_normalized = feature_clips.astype(np.float32) / 255.0
    
    #ensure features have channel dimension
    if len(feat_normalized.shape) == 4:  #(N, T, H, W)
        feat_normalized = feat_normalized[..., np.newaxis]
    
    combined = np.concatenate([rgb_normalized, feat_normalized], axis=-1)
    return combined


def visualize_training_samples(clips, labels, num_samples=4, save_path=None):
    """visualize sample training clips"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))
    
    pos_indices = np.where(np.array(labels) == 1)[0]
    neg_indices = np.where(np.array(labels) == 0)[0]
    
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
            frame = clip[frame_idx]
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            axes[row, col].imshow(frame[..., :3] if frame.shape[-1] > 3 else frame)
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
    from video_scaler import load_scaled_video
    from edge_processor import process_edge_features
    from audio_ground_truth import generate_bump_candidates
    
    video_path = os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4")
    
    if os.path.exists(video_path):
        scaled_path = os.path.join(config.OUTPUT_DIR, "scaled_video.mp4")
        candidates_path = os.path.join(config.OUTPUT_DIR, "bump_candidates.npy")
        
        #load frames
        if not os.path.exists(scaled_path):
            from video_scaler import scale_video
            scale_video(video_path, scaled_path)
        frames = load_scaled_video(scaled_path, max_frames=3000)
        
        #load or generate candidates
        if os.path.exists(candidates_path):
            result = np.load(candidates_path, allow_pickle=True).item()
            candidates = result['candidates']
        else:
            result = generate_bump_candidates(video_path, save_path=candidates_path)
            candidates = result['candidates']
        
        #process features
        feature_frames = process_edge_features(frames)
        
        #create training data
        data = create_training_data(frames, feature_frames, candidates)
        pos_clips, pos_features, pos_info, neg_clips, neg_features, neg_info = data
        
        save_training_data(pos_clips, pos_features, pos_info,
                          neg_clips, neg_features, neg_info)
    else:
        print(f"video not found: {video_path}")
