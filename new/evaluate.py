#evaluation and visualization of bump detection results

import numpy as np
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import config
from model import get_model
from video_scaler import load_scaled_video
from edge_processor import canny_edge_detection
from data_generator import create_combined_features


def load_model(model_path, model_type='simple', device=None):
    """load trained model from checkpoint or weights file"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(model_type, in_channels=4)
    
    #try loading as checkpoint first, then as raw state_dict
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            #assume it's a raw state_dict
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"weights_only load failed: {e}")
        print("trying with weights_only=False...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device


def sliding_window_inference(model, frames, edge_frames, device, 
                             window_size=15, stride=1, batch_size=8):
    """
    run model inference using sliding window over video
    returns bump probability for each frame
    """
    T, H, W, C = frames.shape
    num_windows = (T - window_size) // stride + 1
    
    print(f"running inference on {num_windows} windows...")
    
    #prepare all windows
    predictions = np.zeros(T)
    counts = np.zeros(T)
    
    batch_inputs = []
    batch_positions = []
    
    for i in tqdm(range(num_windows), desc="inference"):
        start = i * stride
        end = start + window_size
        
        #extract window
        clip = frames[start:end]
        edge_clip = edge_frames[start:end]
        
        #combine features
        combined = create_combined_features(
            clip[np.newaxis], edge_clip[np.newaxis]
        )[0]
        
        #transpose to (C, T, H, W)
        combined = np.transpose(combined, (3, 0, 1, 2))
        
        batch_inputs.append(combined)
        batch_positions.append((start, end))
        
        #process batch
        if len(batch_inputs) >= batch_size or i == num_windows - 1:
            batch_tensor = torch.tensor(np.array(batch_inputs), dtype=torch.float32)
            batch_tensor = batch_tensor.to(device)
            
            with torch.no_grad():
                outputs = model(batch_tensor)
            
            probs = outputs.cpu().numpy()
            
            #assign probabilities to frames
            for prob, (s, e) in zip(probs, batch_positions):
                #assign to end of window (prediction point)
                predictions[e-1] += prob
                counts[e-1] += 1
            
            batch_inputs = []
            batch_positions = []
    
    #average overlapping predictions
    mask = counts > 0
    predictions[mask] /= counts[mask]
    
    return predictions


def detect_bumps_from_predictions(predictions, threshold=0.5):
    """
    convert frame-level predictions to bump detections
    simple thresholding - any frame above threshold is a detection
    """
    #find all frames above threshold
    above_threshold = predictions >= threshold
    detected_frames = np.where(above_threshold)[0]
    confidences = predictions[detected_frames]
    
    return detected_frames, confidences


def evaluate_detections(detected_frames, ground_truth_frames, 
                        tolerance_frames=3, total_frames=None):
    """
    evaluate detection performance against ground truth
    """
    detected_set = set(detected_frames)
    gt_set = set(ground_truth_frames)
    
    #true positives: detected bumps that match ground truth (within tolerance)
    tp = 0
    matched_gt = set()
    matched_det = set()
    
    for det in detected_frames:
        for gt in ground_truth_frames:
            if abs(det - gt) <= tolerance_frames and gt not in matched_gt:
                tp += 1
                matched_gt.add(gt)
                matched_det.add(det)
                break
    
    fp = len(detected_frames) - tp
    fn = len(ground_truth_frames) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched_detections': list(matched_det),
        'false_alarms': list(set(detected_frames) - matched_det)
    }


def plot_detection_results(predictions, detected_frames, ground_truth_frames=None,
                          save_path=None, fps=15):
    """
    plot detection results over time
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    time_axis = np.arange(len(predictions)) / fps
    
    #plot predictions
    axes[0].plot(time_axis, predictions, 'b-', alpha=0.7, label='bump probability')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='threshold (0.5)')
    axes[0].set_ylabel('probability')
    axes[0].legend()
    axes[0].set_title('bump detection predictions')
    
    #plot detections
    detection_signal = np.zeros(len(predictions))
    for d in detected_frames:
        if d < len(detection_signal):
            detection_signal[d] = 1
    
    axes[1].stem(time_axis, detection_signal, 'b-', markerfmt='bo', 
                 basefmt='k-', label='detected bumps')
    
    if ground_truth_frames is not None:
        gt_signal = np.zeros(len(predictions))
        for g in ground_truth_frames:
            if g < len(gt_signal):
                gt_signal[g] = 0.8
        axes[1].stem(time_axis, gt_signal, 'g-', markerfmt='g^', 
                     basefmt='k-', label='ground truth')
    
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('bump')
    axes[1].legend()
    axes[1].set_title('detected vs ground truth bumps')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved detection plot to {save_path}")
    
    plt.close(fig)
    return fig


def create_annotated_video(frames, predictions, detected_frames, 
                          ground_truth_frames=None, output_path=None):
    """
    create video with bump detection overlay
    """
    if output_path is None:
        output_path = os.path.join(config.OUTPUT_DIR, 'annotated_video.mp4')
    
    T, H, W, C = frames.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, (W, H + 60))
    
    #convert sets for fast lookup
    detected_set = set(detected_frames)
    gt_set = set(ground_truth_frames) if ground_truth_frames is not None else set()
    
    for t in tqdm(range(T), desc="creating annotated video"):
        frame = frames[t].copy()
        
        #create info bar at bottom
        info_bar = np.zeros((60, W, 3), dtype=np.uint8)
        
        #draw prediction bar
        prob = predictions[t] if t < len(predictions) else 0
        bar_width = int(prob * (W - 20))
        cv2.rectangle(info_bar, (10, 10), (10 + bar_width, 30), (0, 255, 0), -1)
        cv2.rectangle(info_bar, (10, 10), (W - 10, 30), (255, 255, 255), 1)
        
        #add text
        cv2.putText(info_bar, f"prob: {prob:.2f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_bar, f"frame: {t}", (W - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        #highlight bump detections
        if t in detected_set:
            #red border for detected bump
            cv2.rectangle(frame, (0, 0), (W-1, H-1), (255, 0, 0), 3)
            cv2.putText(frame, "BUMP DETECTED!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if t in gt_set:
            #green indicator for ground truth
            cv2.circle(frame, (W - 30, 30), 15, (0, 255, 0), -1)
        
        #combine frame and info bar
        combined = np.vstack([frame, info_bar])
        bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    
    out.release()
    print(f"saved annotated video to {output_path}")


def run_full_evaluation(video_path, model_path, model_type='simple', 
                        ground_truth_path=None, max_frames=5000,
                        scaled_video_path=None):
    """
    run complete evaluation pipeline on a video
    
    scaled_video_path: path to pre-scaled low-res video (skips scaling)
    """
    #load model
    print("loading model...")
    model, device = load_model(model_path, model_type)
    
    #load or process video
    if scaled_video_path and os.path.exists(scaled_video_path):
        print(f"using pre-scaled video: {scaled_video_path}")
        print(f"loading (max {max_frames} frames)...")
        frames = load_scaled_video(scaled_video_path, max_frames=max_frames)
    else:
        scaled_path = os.path.join(config.OUTPUT_DIR, "scaled_video.mp4")
        if os.path.exists(scaled_path):
            print(f"loading scaled video (max {max_frames} frames)...")
            frames = load_scaled_video(scaled_path, max_frames=max_frames)
        else:
            from video_scaler import scale_video
            print("scaling video...")
            scale_video(video_path, scaled_path)
            frames = load_scaled_video(scaled_path, max_frames=max_frames)
    
    #process edges - all canny edges, no filtering
    print("processing edges (all canny edges)...")
    edge_frames = []
    for frame in tqdm(frames, desc="extracting canny edges"):
        edges = canny_edge_detection(frame)
        edge_frames.append(edges)
    edge_frames = np.array(edge_frames)
    
    #run inference
    predictions = sliding_window_inference(model, frames, edge_frames, device)
    
    #detect bumps
    detected_frames, confidences = detect_bumps_from_predictions(
        predictions, threshold=config.DETECTION_THRESHOLD
    )
    print(f"detected {len(detected_frames)} bumps (threshold={config.DETECTION_THRESHOLD})")
    
    #load ground truth if available
    gt_frames = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        print("loading ground truth...")
        gt = np.load(ground_truth_path, allow_pickle=True).item()
        
        #adjust for fps difference
        original_fps = gt['video_fps']
        fps_ratio = config.TARGET_FPS / original_fps
        gt_frames = (gt['bump_frames'] * fps_ratio).astype(int)
        
        #evaluate
        metrics = evaluate_detections(detected_frames, gt_frames, 
                                      total_frames=len(frames))
        print(f"\nevaluation metrics:")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall: {metrics['recall']:.4f}")
        print(f"  f1 score: {metrics['f1']:.4f}")
        print(f"  true positives: {metrics['tp']}")
        print(f"  false positives: {metrics['fp']}")
        print(f"  false negatives: {metrics['fn']}")
    
    #plot results
    plot_path = os.path.join(config.OUTPUT_DIR, 'detection_results.png')
    plot_detection_results(predictions, detected_frames, gt_frames, 
                          save_path=plot_path)
    
    #create annotated video
    video_path = os.path.join(config.OUTPUT_DIR, 'annotated_video.mp4')
    create_annotated_video(frames, predictions, detected_frames, 
                          gt_frames, output_path=video_path)
    
    return predictions, detected_frames, gt_frames


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, 
                       default=os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4"),
                       help='path to original video (for scaling if needed)')
    parser.add_argument('--scaled-video', type=str, default=None,
                       help='path to pre-scaled low-res video (skips scaling)')
    parser.add_argument('--model', type=str, 
                       default=os.path.join(config.MODEL_DIR, "simple_best.pth"))
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['unet', 'simple', 'attention'])
    parser.add_argument('--ground_truth', type=str,
                       default=os.path.join(config.OUTPUT_DIR, "ground_truth.npy"))
    parser.add_argument('--max-frames', type=int, default=5000,
                       help='max frames to evaluate')
    args = parser.parse_args()
    
    run_full_evaluation(
        args.video,
        args.model,
        model_type=args.model_type,
        ground_truth_path=args.ground_truth,
        max_frames=args.max_frames,
        scaled_video_path=args.scaled_video
    )

