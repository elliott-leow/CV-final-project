#generate video with only ML-detected bump-causing features highlighted
import cv2
import numpy as np
from pathlib import Path
import pickle
import joblib

DATA_DIR = Path("data-scaled")
OUTPUT_DIR = Path("output")
FPS = 20
LOOKAHEAD_FRAMES = 15  #frames before bump to show warning

def load_model():
    """load trained bump feature detector"""
    data = joblib.load(OUTPUT_DIR / 'bump_feature_detector.joblib')
    return data['model'], data['scaler'], data['feature_names'], data['grid_size']

def load_bump_detection():
    """load bump detection results"""
    with open(OUTPUT_DIR / 'bump_detection_robust.pkl', 'rb') as f:
        return pickle.load(f)

def extract_region_features(gray, x, y, w, h):
    """extract features for a single region"""
    region = gray[y:y+h, x:x+w]
    if region.size == 0:
        return None
    
    features = {}
    
    features['intensity_mean'] = region.mean()
    features['intensity_std'] = region.std()
    
    edges = cv2.Canny(region, 50, 150)
    features['edge_density'] = edges.mean() / 255
    
    sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    features['grad_x'] = np.abs(sobel_x).mean()
    features['grad_y'] = np.abs(sobel_y).mean()
    features['grad_ratio'] = features['grad_y'] / (features['grad_x'] + 1e-6)
    
    laplacian = cv2.Laplacian(region, cv2.CV_64F)
    features['texture'] = laplacian.var()
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=w//3, maxLineGap=5)
    features['n_lines'] = len(lines) if lines is not None else 0
    
    h_line_score = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 20 or angle > 160:
                h_line_score += 1
    features['h_lines'] = h_line_score
    
    features['contrast'] = region.max() - region.min()
    
    return features

def detect_bump_causing_features(frame, model, scaler, feature_names, grid_size, bump_approaching):
    """detect regions that will cause bumps using trained model"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    roi_start = h // 3
    roi_h = h - roi_start
    
    cell_h = roi_h // grid_size
    cell_w = w // grid_size
    
    detections = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_w
            y = roi_start + row * cell_h
            
            feat = extract_region_features(gray, x, y, cell_w, cell_h)
            if feat is None:
                continue
            
            #add position features
            feat['row'] = row / grid_size
            feat['col'] = col / grid_size
            feat['dist_from_center'] = np.sqrt((row/grid_size - 0.5)**2 + (col/grid_size - 0.5)**2)
            
            #predict
            X = np.array([[feat[k] for k in feature_names]])
            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0, 1]
            
            #only show if bump is approaching AND probability is high
            if bump_approaching and prob > 0.6:
                detections.append({
                    'bbox': (x, y, cell_w, cell_h),
                    'prob': prob,
                    'row': row
                })
    
    return detections

def draw_smart_detections(frame, detections, bump_info=None):
    """draw only bump-causing features"""
    frame_out = frame.copy()
    h, w = frame.shape[:2]
    
    for det in detections:
        x, y, cw, ch = det['bbox']
        prob = det['prob']
        
        #color intensity based on probability
        intensity = int(255 * prob)
        color = (0, intensity, 255)  #orange-red gradient
        
        #draw semi-transparent highlight
        overlay = frame_out.copy()
        cv2.rectangle(overlay, (x, y), (x + cw, y + ch), color, -1)
        alpha = 0.3 * prob
        frame_out = cv2.addWeighted(overlay, alpha, frame_out, 1 - alpha, 0)
        
        #draw border
        cv2.rectangle(frame_out, (x, y), (x + cw, y + ch), color, 2)
        
        #label only for high confidence
        if prob > 0.75:
            cv2.putText(frame_out, f"HAZARD", (x + 2, y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    #warning banner if bump approaching
    if bump_info:
        frames_until, intensity = bump_info
        
        #warning at top
        warning_height = 25
        alpha = min(0.8, intensity / 10)
        
        overlay = frame_out.copy()
        cv2.rectangle(overlay, (0, 0), (w, warning_height), (0, 0, 255), -1)
        frame_out = cv2.addWeighted(overlay, alpha, frame_out, 1 - alpha, 0)
        
        warning_text = f"BUMP IN {frames_until} FRAMES"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame_out, warning_text, (text_x, 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        #pulsing border
        border_width = max(2, int(4 * alpha))
        cv2.rectangle(frame_out, (0, 0), (w-1, h-1), (0, 0, 255), border_width)
    
    return frame_out

def generate_video(video_path, bump_frames, bump_intensities, model, scaler, feature_names, grid_size, output_path):
    """generate annotated video with ML-detected bump features"""
    print("generating smart annotated video...")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    bump_dict = {bf: bi for bf, bi in zip(bump_frames, bump_intensities)}
    
    frame_idx = 0
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #check if bump is approaching
        bump_info = None
        bump_approaching = False
        for bf in bump_frames:
            if 0 < bf - frame_idx <= LOOKAHEAD_FRAMES:
                frames_until = bf - frame_idx
                intensity = bump_dict.get(bf, 5)
                bump_info = (frames_until, intensity)
                bump_approaching = True
                break
        
        #detect bump-causing features (only when bump is approaching)
        detections = detect_bump_causing_features(
            frame, model, scaler, feature_names, grid_size, bump_approaching
        )
        
        #mark bump moment
        if frame_idx in bump_dict:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.putText(frame, "BUMP!", (width//2 - 30, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        #draw annotations
        frame_out = draw_smart_detections(frame, detections, bump_info)
        
        #frame counter
        cv2.putText(frame_out, f"Frame: {frame_idx}", (5, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        out.write(frame_out)
        frame_idx += 1
        
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
    
    cap.release()
    out.release()
    print(f"  saved to {output_path}")

def generate_bump_clips(video_path, bump_frames, bump_intensities, model, scaler, feature_names, grid_size, output_dir):
    """generate clips showing bump-causing features"""
    print("\ngenerating bump clips...")
    
    clips_dir = output_dir / "bump_clips_smart"
    clips_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    bump_dict = {bf: bi for bf, bi in zip(bump_frames, bump_intensities)}
    
    #sort by intensity
    bump_data = sorted(zip(bump_frames, bump_intensities), key=lambda x: x[1], reverse=True)
    
    clip_before = 30
    clip_after = 10
    n_clips = min(15, len(bump_data))
    
    for i, (bf, intensity) in enumerate(bump_data[:n_clips]):
        start = max(0, bf - clip_before)
        end = min(len(frames), bf + clip_after)
        
        clip_path = clips_dir / f"bump_{i+1:02d}_frame{bf}_int{intensity:.1f}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        
        for frame_idx in range(start, end):
            frame = frames[frame_idx].copy()
            frames_until = bf - frame_idx
            
            #detect features (always show in clips)
            bump_approaching = frames_until > 0
            detections = detect_bump_causing_features(
                frame, model, scaler, feature_names, grid_size, bump_approaching=True
            )
            
            if frames_until > 0:
                bump_info = (frames_until, intensity)
            else:
                bump_info = None
            
            if frame_idx == bf:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                cv2.putText(frame, "BUMP!", (width//2 - 30, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            frame_out = draw_smart_detections(frame, detections, bump_info)
            
            #clip info
            status = "BUMP" if frame_idx == bf else (f"{frames_until} to bump" if frames_until > 0 else f"{-frames_until} after")
            cv2.putText(frame_out, f"Bump #{i+1} | {status}", (5, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            out.write(frame_out)
        
        out.release()
        print(f"  clip {i+1}/{n_clips}: frame {bf}, intensity {intensity:.1f}")
    
    print(f"  saved {n_clips} clips to {clips_dir}")

def main():
    #load model
    print("loading model...")
    model, scaler, feature_names, grid_size = load_model()
    
    #load bump detection
    print("loading bump detection...")
    detection = load_bump_detection()
    bump_frames = detection['bump_frames']
    bump_intensities = detection['bump_intensities']
    print(f"  {len(bump_frames)} bumps")
    
    #find video
    video_files = list(DATA_DIR.glob("*.mp4"))
    video_path = video_files[0]
    print(f"video: {video_path.name}")
    
    #generate full video
    generate_video(
        video_path, bump_frames, bump_intensities,
        model, scaler, feature_names, grid_size,
        OUTPUT_DIR / 'bump_annotated_smart.mp4'
    )
    
    #generate clips
    generate_bump_clips(
        video_path, bump_frames, bump_intensities,
        model, scaler, feature_names, grid_size,
        OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\noutput:")
    print(f"  - {OUTPUT_DIR / 'bump_annotated_smart.mp4'}")
    print(f"  - {OUTPUT_DIR / 'bump_clips_smart/'}")

if __name__ == "__main__":
    main()




