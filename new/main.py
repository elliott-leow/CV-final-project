#main pipeline script for bump detection with checkpoint system
#orchestrates all components with ability to resume from checkpoints

import os
import argparse
import numpy as np
from tqdm import tqdm
import json

import config
from audio_ground_truth import generate_ground_truth, visualize_ground_truth
from video_scaler import scale_video, load_scaled_video, get_frame_count
from edge_processor import process_edges, canny_edge_detection, create_feature_video, create_overlay_video
from data_generator import save_training_data, visualize_training_samples
from train import train_model, plot_training_history
from evaluate import run_full_evaluation


#checkpoint definitions
CHECKPOINTS = {
    0: 'start',
    1: 'ground_truth',
    2: 'video_scaling', 
    3: 'edge_processing',
    4: 'training_data',
    5: 'model_training',
    6: 'evaluation'
}


def get_checkpoint_file():
    return os.path.join(config.OUTPUT_DIR, 'checkpoint.json')


def save_checkpoint(step, data=None):
    """save checkpoint progress"""
    checkpoint = {
        'step': step,
        'step_name': CHECKPOINTS.get(step, 'unknown'),
        'data': data or {}
    }
    with open(get_checkpoint_file(), 'w') as f:
        json.dump(checkpoint, f)
    print(f"  [checkpoint saved: step {step} - {CHECKPOINTS.get(step)}]")


def load_checkpoint():
    """load checkpoint if exists"""
    cp_file = get_checkpoint_file()
    if os.path.exists(cp_file):
        with open(cp_file, 'r') as f:
            return json.load(f)
    return None


def create_training_clips_from_data(frames, edge_frames, bump_frames,
                                     frames_before=5, clip_length=15, 
                                     min_frames_to_next=30):
    """create training clips from processed data"""
    total_frames = len(frames)
    
    clips = []
    edge_clips = []
    labels = []
    
    print("creating positive samples...")
    for bump_frame in tqdm(bump_frames, desc="positive clips"):
        start_frame = bump_frame - frames_before
        end_frame = start_frame + clip_length
        
        if start_frame < 0 or end_frame > total_frames:
            continue
        
        clips.append(frames[start_frame:end_frame])
        edge_clips.append(edge_frames[start_frame:end_frame])
        labels.append(1)
    
    num_pos = len(clips)
    print(f"  created {num_pos} positive clips")
    
    print("creating negative samples...")
    bump_frames_sorted = np.sort(bump_frames)
    
    negative_starts = []
    for frame_idx in range(0, total_frames - clip_length, 10):
        end_frame = frame_idx + clip_length
        
        has_bump = np.any((bump_frames_sorted >= frame_idx) & (bump_frames_sorted < end_frame))
        if has_bump:
            continue
        
        future_bumps = bump_frames_sorted[bump_frames_sorted >= end_frame]
        if len(future_bumps) > 0:
            dist_to_next = future_bumps[0] - end_frame
            if dist_to_next >= min_frames_to_next:
                negative_starts.append(frame_idx)
        else:
            negative_starts.append(frame_idx)
    
    num_neg_needed = min(num_pos * 2, len(negative_starts))
    if len(negative_starts) > 0:
        step = max(1, len(negative_starts) // num_neg_needed)
        sampled = negative_starts[::step][:num_neg_needed]
        
        for start_frame in tqdm(sampled, desc="negative clips"):
            end_frame = start_frame + clip_length
            clips.append(frames[start_frame:end_frame])
            edge_clips.append(edge_frames[start_frame:end_frame])
            labels.append(0)
    
    print(f"created {num_pos} positive and {len(labels) - num_pos} negative clips")
    
    return np.array(clips), np.array(edge_clips), np.array(labels)


def run_pipeline(video_path, model_type='simple', epochs=None, 
                max_frames=5000, from_checkpoint=None, fresh_start=False):
    """
    run complete bump detection pipeline with checkpoint support
    
    from_checkpoint: int or str - start from this checkpoint (0-6 or name)
    fresh_start: bool - ignore all checkpoints and start fresh
    """
    epochs = epochs or config.NUM_EPOCHS
    
    print("="*60)
    print("BUMP DETECTION PIPELINE")
    print("="*60)
    
    #determine starting checkpoint
    start_step = 0
    if fresh_start:
        print("fresh start - ignoring all checkpoints")
        start_step = 0
    elif from_checkpoint is not None:
        if isinstance(from_checkpoint, str):
            #find step by name
            for step, name in CHECKPOINTS.items():
                if name == from_checkpoint:
                    start_step = step
                    break
        else:
            start_step = int(from_checkpoint)
        print(f"starting from checkpoint {start_step}: {CHECKPOINTS.get(start_step)}")
    else:
        #check for saved checkpoint
        saved = load_checkpoint()
        if saved:
            start_step = saved['step']
            print(f"resuming from checkpoint {start_step}: {CHECKPOINTS.get(start_step)}")
    
    #file paths
    gt_path = os.path.join(config.OUTPUT_DIR, "ground_truth.npy")
    scaled_path = os.path.join(config.OUTPUT_DIR, "scaled_video.mp4")
    edges_path = os.path.join(config.OUTPUT_DIR, "edges.npy")
    feature_path = os.path.join(config.OUTPUT_DIR, "feature_map.mp4")
    overlay_path = os.path.join(config.OUTPUT_DIR, "edge_overlay.mp4")
    model_path = os.path.join(config.MODEL_DIR, f"{model_type}_best.pth")
    
    #========== STEP 1: GROUND TRUTH ==========
    if start_step <= 1:
        print("\n" + "="*60)
        print("STEP 1: GROUND TRUTH GENERATION (>4kHz audio)")
        print("="*60)
        
        gt = generate_ground_truth(video_path, save_path=gt_path)
        viz_path = os.path.join(config.OUTPUT_DIR, "ground_truth_viz.png")
        visualize_ground_truth(gt, save_path=viz_path)
        
        print(f"detected {len(gt['bump_frames'])} bumps from audio")
        save_checkpoint(1)
    else:
        print("\n[skipping step 1: ground truth - loading from file]")
        gt = np.load(gt_path, allow_pickle=True).item()
        print(f"loaded {len(gt['bump_frames'])} bumps")
    
    #========== STEP 2: VIDEO SCALING ==========
    if start_step <= 2:
        print("\n" + "="*60)
        print("STEP 2: VIDEO SCALING")
        print("="*60)
        
        total_frames, _ = scale_video(video_path, scaled_path)
        save_checkpoint(2, {'total_frames': total_frames})
    else:
        print("\n[skipping step 2: video scaling - using existing file]")
        total_frames = get_frame_count(scaled_path)
        print(f"scaled video has {total_frames} frames")
    
    #limit frames for memory
    use_frames = min(total_frames, max_frames)
    print(f"using {use_frames} of {total_frames} frames")
    
    #========== STEP 3: EDGE PROCESSING ==========
    if start_step <= 3:
        print("\n" + "="*60)
        print("STEP 3: EDGE DETECTION (all canny edges, no filtering)")
        print("="*60)
        
        print(f"loading {use_frames} frames...")
        frames = load_scaled_video(scaled_path, max_frames=use_frames)
        
        print("processing edges...")
        edge_results = process_edges(frames)
        edge_frames = edge_results['filtered']
        
        #save edges
        print("saving edge data...")
        np.save(edges_path, edge_frames)
        
        #save videos (subset for preview)
        print("saving preview videos...")
        subset_size = min(1000, len(frames))
        create_feature_video(frames[:subset_size], edge_frames[:subset_size], feature_path)
        create_overlay_video(frames[:subset_size], edge_frames[:subset_size], overlay_path)
        
        save_checkpoint(3)
    else:
        print("\n[skipping step 3: edge processing - loading from file]")
        frames = load_scaled_video(scaled_path, max_frames=use_frames)
        edge_frames = np.load(edges_path)
        if len(edge_frames) > use_frames:
            edge_frames = edge_frames[:use_frames]
        print(f"loaded {len(edge_frames)} edge frames")
    
    #========== STEP 4: TRAINING DATA ==========
    if start_step <= 4:
        print("\n" + "="*60)
        print("STEP 4: TRAINING DATA GENERATION")
        print("="*60)
        
        #adjust bump frames for new fps
        original_fps = gt['video_fps']
        fps_ratio = config.TARGET_FPS / original_fps
        bump_frames_scaled = (gt['bump_frames'] * fps_ratio).astype(int)
        
        #filter to only bumps within our frame range
        bump_frames_valid = bump_frames_scaled[bump_frames_scaled < use_frames]
        print(f"using {len(bump_frames_valid)} bumps within frame range")
        
        clips, edge_clips, labels = create_training_clips_from_data(
            frames, edge_frames, bump_frames_valid
        )
        
        #save training data
        clip_info = []
        save_training_data(clips, edge_clips, labels, clip_info)
        
        #visualize samples
        if len(clips) > 0:
            viz_path = os.path.join(config.OUTPUT_DIR, "training_samples.png")
            visualize_training_samples(clips, labels, save_path=viz_path)
        
        save_checkpoint(4)
    else:
        print("\n[skipping step 4: training data - using existing files]")
    
    #free memory
    del frames, edge_frames
    
    #========== STEP 5: MODEL TRAINING ==========
    if start_step <= 5:
        print("\n" + "="*60)
        print("STEP 5: MODEL TRAINING")
        print("="*60)
        
        model, history = train_model(model_type=model_type, epochs=epochs)
        
        plot_path = os.path.join(config.OUTPUT_DIR, f'training_history_{model_type}.png')
        plot_training_history(history, save_path=plot_path)
        
        save_checkpoint(5)
    else:
        print("\n[skipping step 5: model training - using existing model]")
    
    #========== STEP 6: EVALUATION ==========
    print("\n" + "="*60)
    print("STEP 6: EVALUATION")
    print("="*60)
    
    if os.path.exists(model_path):
        run_full_evaluation(
            video_path,
            model_path,
            model_type=model_type,
            ground_truth_path=gt_path,
            max_frames=max_frames
        )
    else:
        print(f"model not found: {model_path}")
    
    save_checkpoint(6)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\ncheckpoint steps:")
    for step, name in CHECKPOINTS.items():
        print(f"  {step}: {name}")
    print(f"\nto resume from a checkpoint, use: --from-checkpoint <step>")


def main():
    parser = argparse.ArgumentParser(description='Bump Detection Pipeline')
    parser.add_argument('--video', type=str, 
                       default=os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4"),
                       help='path to input video')
    parser.add_argument('--model', type=str, default='simple',
                       choices=['unet', 'simple', 'attention'],
                       help='model architecture')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                       help='number of training epochs')
    parser.add_argument('--max-frames', type=int, default=5000,
                       help='max frames to use for training')
    parser.add_argument('--from-checkpoint', type=str, default=None,
                       help='start from checkpoint (0-6 or name: start, ground_truth, video_scaling, edge_processing, training_data, model_training, evaluation)')
    parser.add_argument('--fresh', action='store_true',
                       help='ignore all checkpoints and start fresh')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"error: video not found: {args.video}")
        return
    
    #convert checkpoint arg
    from_cp = None
    if args.from_checkpoint:
        try:
            from_cp = int(args.from_checkpoint)
        except ValueError:
            from_cp = args.from_checkpoint
    
    run_pipeline(
        video_path=args.video,
        model_type=args.model,
        epochs=args.epochs,
        max_frames=args.max_frames,
        from_checkpoint=from_cp,
        fresh_start=args.fresh
    )


if __name__ == "__main__":
    main()
