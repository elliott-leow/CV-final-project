#step 8: sliding window inference with non-maximum suppression
#runs trained model on full video and generates bump detection plot

import numpy as np
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import config
from model import get_model
from video_scaler import load_scaled_video, load_video_chunk, get_frame_count, scale_video
from edge_processor import process_frame_edges


def load_model(model_path, model_type='unet', device=None):
    """load trained model from checkpoint or weights file"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(model_type, in_channels=4)
    
    #try loading as checkpoint, then as raw state_dict
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
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


def prepare_clip_input(clip, feature_clip):
    """prepare a single clip for model input"""
    clip_norm = clip.astype(np.float32) / 255.0
    feat_norm = feature_clip.astype(np.float32) / 255.0
    
    #use single edge channel
    if len(feat_norm.shape) == 4:
        feat_norm = feat_norm[..., :1]
    elif len(feat_norm.shape) == 3:
        feat_norm = feat_norm[..., np.newaxis]
    
    x = np.concatenate([clip_norm, feat_norm], axis=-1)
    x = np.transpose(x, (3, 0, 1, 2))  #(C, T, H, W)
    
    return x


def sliding_window_inference(model, video_path, device,
                             window_size=None, stride=None, batch_size=8,
                             max_frames=None):
    """
    step 8: run trained model on full video using sliding window
    
    for each window starting at time s:
        build input tensor X_s
        compute p_θ(y=1|X_s)
        predicted bump time = s + H (early warning horizon)
    
    returns:
        predictions: bump probability for each frame
        bump_times: predicted bump frame indices (from clip start + H)
    """
    window_size = window_size or config.CLIP_LENGTH
    stride = stride or config.INFERENCE_STRIDE
    horizon = config.EARLY_WARNING_HORIZON
    
    total_frames = get_frame_count(video_path)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    num_windows = (total_frames - window_size) // stride + 1
    
    print(f"running sliding window inference...")
    print(f"  window size: {window_size}, stride: {stride}")
    print(f"  total frames: {total_frames}, windows: {num_windows}")
    
    #predictions for each window (indexed by clip start)
    window_probs = np.zeros(num_windows)
    window_starts = np.zeros(num_windows, dtype=int)
    
    #process in batches
    batch_clips = []
    batch_features = []
    batch_indices = []
    
    for i in tqdm(range(num_windows), desc="inference"):
        start = i * stride
        
        #load clip
        clip = load_video_chunk(video_path, start, window_size)
        if clip is None or len(clip) < window_size:
            continue
        
        #compute edge features
        features = np.array([process_frame_edges(f) for f in clip])
        
        #prepare input
        x = prepare_clip_input(clip, features)
        
        batch_clips.append(x)
        batch_indices.append(i)
        window_starts[i] = start
        
        #process batch
        if len(batch_clips) >= batch_size or i == num_windows - 1:
            if len(batch_clips) > 0:
                batch_tensor = torch.tensor(np.array(batch_clips), dtype=torch.float32)
                batch_tensor = batch_tensor.to(device)
                
                with torch.no_grad():
                    probs = model.predict_proba(batch_tensor)
                
                probs = probs.cpu().numpy()
                
                for j, idx in enumerate(batch_indices):
                    window_probs[idx] = probs[j]
                
                batch_clips = []
                batch_indices = []
    
    #convert to per-frame predictions
    #each window predicts bump at start + horizon
    frame_predictions = np.zeros(total_frames)
    frame_counts = np.zeros(total_frames)
    
    for i in range(num_windows):
        predicted_bump_frame = window_starts[i] + horizon
        if predicted_bump_frame < total_frames:
            frame_predictions[predicted_bump_frame] += window_probs[i]
            frame_counts[predicted_bump_frame] += 1
    
    #average overlapping predictions
    mask = frame_counts > 0
    frame_predictions[mask] /= frame_counts[mask]
    
    return frame_predictions, window_probs, window_starts


def non_maximum_suppression(predictions, threshold=None, window=None):
    """
    apply temporal non-maximum suppression to predictions
    
    for high probability responses within a window, keep only the highest
    
    returns:
        detected_frames: frame indices of detected bumps
        confidences: confidence scores for detections
    """
    threshold = threshold or config.DETECTION_THRESHOLD
    window = window or config.NMS_WINDOW
    
    #find frames above threshold
    candidates = np.where(predictions >= threshold)[0]
    
    if len(candidates) == 0:
        return np.array([]), np.array([])
    
    #sort by confidence (descending)
    sorted_idx = np.argsort(predictions[candidates])[::-1]
    candidates = candidates[sorted_idx]
    
    #apply NMS
    keep = []
    suppressed = set()
    
    for c in candidates:
        if c in suppressed:
            continue
        
        keep.append(c)
        
        #suppress nearby frames
        for delta in range(-window, window + 1):
            suppressed.add(c + delta)
    
    detected_frames = np.array(keep)
    confidences = predictions[detected_frames]
    
    #sort by frame index
    sort_idx = np.argsort(detected_frames)
    detected_frames = detected_frames[sort_idx]
    confidences = confidences[sort_idx]
    
    return detected_frames, confidences


def evaluate_detections(detected_frames, ground_truth_frames,
                        tolerance_frames=None):
    """evaluate detection performance against ground truth"""
    tolerance_frames = tolerance_frames or config.EVAL_TOLERANCE_FRAMES
    if len(detected_frames) == 0:
        return {
            'tp': 0, 'fp': 0, 'fn': len(ground_truth_frames),
            'precision': 0, 'recall': 0, 'f1': 0,
            'matched_detections': [], 'false_alarms': []
        }
    
    #match detections to ground truth
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
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'matched_detections': list(matched_det),
        'false_alarms': list(set(detected_frames) - matched_det)
    }


def plot_detection_results(predictions, detected_frames, 
                          ground_truth_frames=None, audio_candidates=None,
                          save_path=None, fps=None):
    """
    generate bump detection plot (step 8)
    
    x-axis: time
    y-axis: model bump probability / thresholded detections
    overlay: audio candidates, ground truth, detected bumps
    """
    fps = fps or config.TARGET_FPS
    time_axis = np.arange(len(predictions)) / fps
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    #plot 1: raw predictions
    axes[0].plot(time_axis, predictions, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].axhline(y=config.DETECTION_THRESHOLD, color='r', linestyle='--', 
                    alpha=0.5, label=f'threshold ({config.DETECTION_THRESHOLD})')
    axes[0].fill_between(time_axis, 0, predictions, alpha=0.3)
    axes[0].set_ylabel('bump probability')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title('model predictions (sliding window)')
    
    #plot 2: detections vs ground truth
    det_signal = np.zeros(len(predictions))
    for d in detected_frames:
        if d < len(det_signal):
            det_signal[d] = 1
    
    markerline, stemlines, baseline = axes[1].stem(
        time_axis, det_signal, 'b-', markerfmt='bo', basefmt='k-'
    )
    stemlines.set_alpha(0.7)
    markerline.set_markersize(4)
    
    if ground_truth_frames is not None:
        gt_signal = np.zeros(len(predictions))
        for g in ground_truth_frames:
            if g < len(gt_signal):
                gt_signal[g] = 0.8
        
        markerline2, stemlines2, _ = axes[1].stem(
            time_axis, gt_signal, 'g-', markerfmt='g^', basefmt='k-'
        )
        stemlines2.set_alpha(0.5)
        markerline2.set_markersize(4)
        axes[1].plot([], [], 'g^', label='ground truth')
    
    axes[1].plot([], [], 'bo', label='detected')
    axes[1].set_ylabel('detection')
    axes[1].set_ylim(0, 1.2)
    axes[1].legend()
    axes[1].set_title('detected bumps vs ground truth')
    
    #plot 3: audio candidates overlay
    if audio_candidates is not None:
        for c in audio_candidates:
            t = c['audio_frame'] / fps
            axes[2].axvline(x=t, color='orange', alpha=0.5, linewidth=1)
            
            #show window
            window_times = c['window'] / fps
            axes[2].axvspan(window_times.min(), window_times.max(), 
                           alpha=0.1, color='orange')
    
    #overlay predictions
    axes[2].plot(time_axis, predictions, 'b-', alpha=0.5, linewidth=0.5)
    
    #mark detections
    for d in detected_frames:
        if d < len(time_axis):
            axes[2].axvline(x=time_axis[d], color='blue', alpha=0.7, linewidth=1)
    
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel('probability')
    axes[2].set_title('audio candidates (orange) vs model detections (blue)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved detection plot to {save_path}")
    
    plt.close(fig)
    return fig


def create_annotated_video(video_path, predictions, detected_frames,
                          ground_truth_frames=None, output_path=None,
                          max_frames=None):
    """create video with bump detection overlay"""
    if output_path is None:
        output_path = os.path.join(config.OUTPUT_DIR, 'annotated_video.mp4')
    
    total_frames = get_frame_count(video_path)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    #get video dimensions
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, (W, H + 60))
    
    detected_set = set(detected_frames)
    gt_set = set(ground_truth_frames) if ground_truth_frames is not None else set()
    
    cap = cv2.VideoCapture(video_path)
    
    for t in tqdm(range(total_frames), desc="creating annotated video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        #info bar
        info_bar = np.zeros((60, W, 3), dtype=np.uint8)
        
        #probability bar
        prob = predictions[t] if t < len(predictions) else 0
        bar_width = int(prob * (W - 20))
        color = (0, 255, 0) if prob < 0.5 else (0, 165, 255) if prob < 0.7 else (0, 0, 255)
        cv2.rectangle(info_bar, (10, 10), (10 + bar_width, 30), color, -1)
        cv2.rectangle(info_bar, (10, 10), (W - 10, 30), (255, 255, 255), 1)
        
        #text
        cv2.putText(info_bar, f"P(bump): {prob:.4f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_bar, f"frame: {t}", (W - 120, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        #detection highlight
        if t in detected_set:
            cv2.rectangle(frame, (0, 0), (W-1, H-1), (0, 0, 255), 4)
            cv2.putText(frame, "BUMP!", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        #ground truth indicator
        if t in gt_set:
            cv2.circle(frame, (W - 30, 30), 15, (0, 255, 0), -1)
        
        combined = np.vstack([frame, info_bar])
        out.write(combined)
    
    cap.release()
    out.release()
    print(f"saved annotated video to {output_path}")


def run_full_evaluation(video_path, model_path, model_type='unet',
                        ground_truth_path=None, max_frames=5000,
                        scaled_video_path=None):
    """
    step 8: run complete inference and evaluation pipeline
    """
    #auto-generate scaled video path if not provided
    if scaled_video_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        scaled_video_path = os.path.join(config.OUTPUT_DIR, f"{base}_scaled.mp4")
    
    #scale video if it doesn't exist
    if not os.path.exists(scaled_video_path):
        print(f"scaling video to {config.TARGET_WIDTH}x{config.TARGET_HEIGHT} @ {config.TARGET_FPS}fps...")
        scale_video(video_path, scaled_video_path)
    
    eval_video = scaled_video_path
    print(f"using scaled video: {eval_video}")
    
    #load model
    print("loading model...")
    model, device = load_model(model_path, model_type)
    
    #run inference
    predictions, window_probs, window_starts = sliding_window_inference(
        model, eval_video, device, max_frames=max_frames
    )
    
    #apply NMS
    detected_frames, confidences = non_maximum_suppression(predictions)
    print(f"detected {len(detected_frames)} bumps after NMS")
    
    #load ground truth if available
    gt_frames = None
    audio_candidates = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        print("loading ground truth/candidates...")
        gt = np.load(ground_truth_path, allow_pickle=True).item()
        
        #check if this is new-style candidates or old-style ground truth
        if 'candidates' in gt:
            audio_candidates = gt['candidates']
            #use audio bump frames as ground truth for evaluation
            gt_frames = gt['bump_frames']
        else:
            gt_frames = gt.get('bump_frames', None)
            if gt_frames is not None:
                #adjust for fps difference
                original_fps = gt['video_fps']
                fps_ratio = config.TARGET_FPS / original_fps
                gt_frames = (gt_frames * fps_ratio).astype(int)
        
        #evaluate
        if gt_frames is not None:
            metrics = evaluate_detections(detected_frames, gt_frames)
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
                          audio_candidates, save_path=plot_path)
    
    #create annotated video
    video_out = os.path.join(config.OUTPUT_DIR, 'annotated_video.mp4')
    create_annotated_video(eval_video, predictions, detected_frames,
                          gt_frames, output_path=video_out, max_frames=max_frames)
    
    return predictions, detected_frames, gt_frames


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                       default=os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4"))
    parser.add_argument('--scaled-video', type=str, default=None)
    parser.add_argument('--model', type=str,
                       default=os.path.join(config.MODEL_DIR, "resnet_gru_best.pth"))
    parser.add_argument('--model_type', type=str, default='resnet_gru',
                       choices=['resnet_gru', 'resnet_cnn', 'resnet_lstm',
                                'efficientnet_gru', 'mobilenet_gru',
                                'lightweight_gru', 'lightweight_cnn'])
    parser.add_argument('--ground_truth', type=str,
                       default=os.path.join(config.OUTPUT_DIR, "bump_candidates.npy"))
    parser.add_argument('--max-frames', type=int, default=5000)
    args = parser.parse_args()
    
    run_full_evaluation(
        args.video,
        args.model,
        model_type=args.model_type,
        ground_truth_path=args.ground_truth,
        max_frames=args.max_frames,
        scaled_video_path=args.scaled_video
    )
