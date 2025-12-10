#main pipeline script for bump detection with EM training
#orchestrates all steps 0-8 with checkpoint support

import os
import argparse
import numpy as np
from tqdm import tqdm
import json

import config


#checkpoint definitions
CHECKPOINTS = {
    0: 'start',
    1: 'video_scaling',       #step 0
    2: 'audio_candidates',    #steps 1-2
    3: 'edge_processing',     #step 4 (preprocessing)
    4: 'training_data',       #step 3
    5: 'model_training',      #steps 5-7
    6: 'evaluation'           #step 8
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


def run_pipeline(video_path, model_type='unet', 
                 em_iterations=None, epochs_per_m=None,
                 max_frames=5000, from_checkpoint=None, fresh_start=False,
                 scaled_video_path=None):
    """
    run complete bump detection pipeline with EM training
    
    step 0: downscale video with edge preservation
    steps 1-2: audio bump candidates with gaussian priors
    step 3: clip extraction with noisy labels
    step 4: edge detection and feature stacking
    steps 5-7: em-style training
    step 8: sliding window inference with nms
    """
    em_iterations = em_iterations or config.EM_ITERATIONS
    epochs_per_m = epochs_per_m or config.EM_EPOCHS_PER_M_STEP
    
    print("="*60)
    print("BUMP DETECTION PIPELINE (EM Training)")
    print("="*60)
    
    #imports
    from video_scaler import scale_video, load_scaled_video, get_frame_count
    from audio_ground_truth import generate_bump_candidates, visualize_bump_candidates
    from edge_processor import process_edge_features, create_feature_video, create_overlay_video
    from data_generator import create_training_data, save_training_data, visualize_training_samples
    from train import train_em, plot_training_history
    from evaluate import run_full_evaluation
    
    #check if using pre-scaled video
    use_prescaled = scaled_video_path is not None
    if use_prescaled:
        print(f"using pre-scaled video: {scaled_video_path}")
    
    #determine starting checkpoint
    start_step = 0
    if fresh_start:
        print("fresh start - ignoring all checkpoints")
    elif from_checkpoint is not None:
        if isinstance(from_checkpoint, str):
            for step, name in CHECKPOINTS.items():
                if name == from_checkpoint:
                    start_step = step
                    break
        else:
            start_step = int(from_checkpoint)
        print(f"starting from checkpoint {start_step}: {CHECKPOINTS.get(start_step)}")
    else:
        saved = load_checkpoint()
        if saved:
            start_step = saved['step']
            print(f"resuming from checkpoint {start_step}: {CHECKPOINTS.get(start_step)}")
    
    #file paths
    scaled_path = scaled_video_path if use_prescaled else os.path.join(config.OUTPUT_DIR, "scaled_video.mp4")
    candidates_path = os.path.join(config.OUTPUT_DIR, "bump_candidates.npy")
    #model path uses model_type directly
    model_path = os.path.join(config.MODEL_DIR, f"{model_type}_best.pth")
    
    #========== STEP 0: VIDEO SCALING ==========
    if use_prescaled:
        print("\n" + "="*60)
        print("STEP 0: VIDEO SCALING (using pre-scaled video)")
        print("="*60)
        total_frames = get_frame_count(scaled_path)
        print(f"pre-scaled video: {scaled_path}")
        print(f"total frames: {total_frames}")
        save_checkpoint(1, {'total_frames': total_frames})
    elif start_step <= 1:
        print("\n" + "="*60)
        print("STEP 0: VIDEO SCALING (edge-preserving downscale)")
        print("="*60)
        
        total_frames, _ = scale_video(video_path, scaled_path, keep_audio=True)
        save_checkpoint(1, {'total_frames': total_frames})
    else:
        print("\n[skipping step 0: video scaling - using existing file]")
        total_frames = get_frame_count(scaled_path)
        print(f"scaled video has {total_frames} frames")
    
    use_frames = min(total_frames, max_frames)
    print(f"using {use_frames} of {total_frames} frames")
    
    #========== STEPS 1-2: AUDIO BUMP CANDIDATES ==========
    if start_step <= 2:
        print("\n" + "="*60)
        print("STEPS 1-2: AUDIO BUMP CANDIDATES WITH GAUSSIAN PRIORS")
        print("="*60)
        
        result = generate_bump_candidates(video_path, save_path=candidates_path)
        viz_path = os.path.join(config.OUTPUT_DIR, "bump_candidates_viz.png")
        visualize_bump_candidates(result, save_path=viz_path)
        
        print(f"detected {len(result['bump_frames'])} audio bump candidates")
        candidates = result['candidates']
        save_checkpoint(2)
    else:
        print("\n[skipping steps 1-2: audio candidates - loading from file]")
        result = np.load(candidates_path, allow_pickle=True).item()
        candidates = result['candidates']
        print(f"loaded {len(candidates)} candidates")
    
    #filter candidates to frame range
    valid_candidates = [c for c in candidates if c['audio_frame'] < use_frames]
    print(f"using {len(valid_candidates)} candidates within frame range")
    
    #========== STEP 4: EDGE PROCESSING (before step 3 for features) ==========
    if start_step <= 3:
        print("\n" + "="*60)
        print("STEP 4: EDGE DETECTION AND FEATURE STACKING")
        print("="*60)
        
        print(f"loading {use_frames} frames...")
        frames = load_scaled_video(scaled_path, max_frames=use_frames)
        
        print("processing edge features...")
        features = process_edge_features(frames)
        
        #save preview videos
        print("saving preview videos...")
        subset = min(1000, len(frames))
        feature_path = os.path.join(config.OUTPUT_DIR, "feature_map.mp4")
        create_feature_video(frames[:subset], features[:subset], feature_path)
        
        overlay_path = os.path.join(config.OUTPUT_DIR, "edge_overlay.mp4")
        create_overlay_video(frames[:subset], features[:subset], overlay_path)
        
        save_checkpoint(3)
    else:
        print("\n[skipping step 4: edge processing - loading frames]")
        frames = load_scaled_video(scaled_path, max_frames=use_frames)
        features = process_edge_features(frames, show_progress=True)
    
    #========== STEP 3: TRAINING DATA GENERATION ==========
    if start_step <= 4:
        print("\n" + "="*60)
        print("STEP 3: CLIP EXTRACTION WITH NOISY LABELS")
        print("="*60)
        
        data = create_training_data(frames, features, valid_candidates)
        pos_clips, pos_features, pos_info, neg_clips, neg_features, neg_info = data
        
        save_training_data(pos_clips, pos_features, pos_info,
                          neg_clips, neg_features, neg_info)
        
        #visualize samples
        if len(pos_clips) > 0:
            all_clips = np.concatenate([pos_clips[:10], neg_clips[:10]])
            all_labels = np.concatenate([np.ones(min(10, len(pos_clips))), 
                                        np.zeros(min(10, len(neg_clips)))])
            viz_path = os.path.join(config.OUTPUT_DIR, "training_samples.png")
            visualize_training_samples(all_clips, all_labels, save_path=viz_path)
        
        save_checkpoint(4)
    else:
        print("\n[skipping step 3: training data - using existing files]")
    
    #free memory
    del frames, features
    
    #========== STEPS 5-7: EM TRAINING ==========
    if start_step <= 5:
        print("\n" + "="*60)
        print("STEPS 5-7: EM-STYLE TRAINING")
        print("="*60)
        
        model, history = train_em(
            model_type=model_type,
            em_iterations=em_iterations,
            epochs_per_m=epochs_per_m
        )
        
        plot_path = os.path.join(config.OUTPUT_DIR, f'training_history_{model_type}.png')
        plot_training_history(history, save_path=plot_path)
        
        save_checkpoint(5)
    else:
        print("\n[skipping steps 5-7: model training - using existing model]")
    
    #========== STEP 8: EVALUATION ==========
    print("\n" + "="*60)
    print("STEP 8: SLIDING WINDOW INFERENCE AND EVALUATION")
    print("="*60)
    
    if os.path.exists(model_path):
        run_full_evaluation(
            video_path,
            model_path,
            model_type=model_type,
            ground_truth_path=candidates_path,
            max_frames=max_frames,
            scaled_video_path=scaled_path
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
    parser = argparse.ArgumentParser(description='Bump Detection Pipeline with EM Training')
    parser.add_argument('--video', type=str,
                       default=os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4"),
                       help='path to input video')
    parser.add_argument('--scaled-video', type=str, default=None,
                       help='path to pre-scaled video (skips scaling step)')
    parser.add_argument('--model', type=str, default='resnet_gru',
                       choices=['resnet_gru', 'resnet_cnn', 'resnet_lstm',
                                'efficientnet_gru', 'mobilenet_gru',
                                'lightweight_gru', 'lightweight_cnn'],
                       help='model architecture (pretrained encoder + temporal head)')
    parser.add_argument('--em-iterations', type=int, default=config.EM_ITERATIONS,
                       help='number of EM iterations')
    parser.add_argument('--epochs-per-m', type=int, default=config.EM_EPOCHS_PER_M_STEP,
                       help='epochs per M-step')
    parser.add_argument('--max-frames', type=int, default=5000,
                       help='max frames to use for training')
    parser.add_argument('--from-checkpoint', type=str, default=None,
                       help='start from checkpoint (0-6 or name)')
    parser.add_argument('--fresh', action='store_true',
                       help='ignore all checkpoints and start fresh')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"error: video not found: {args.video}")
        return
    
    if args.scaled_video and not os.path.exists(args.scaled_video):
        print(f"error: scaled video not found: {args.scaled_video}")
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
        em_iterations=args.em_iterations,
        epochs_per_m=args.epochs_per_m,
        max_frames=args.max_frames,
        from_checkpoint=from_cp,
        fresh_start=args.fresh,
        scaled_video_path=args.scaled_video
    )


if __name__ == "__main__":
    main()
