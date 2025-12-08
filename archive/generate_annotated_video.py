#generate annotated video with bump-causing objects labeled
import cv2
import numpy as np
from pathlib import Path
import pickle
from scipy import ndimage

DATA_DIR = Path("data-scaled")
OUTPUT_DIR = Path("output")
FPS = 20
LOOKAHEAD_FRAMES = 10  #frames before bump to start warning

def load_bump_detection():
    """load saved bump detection results"""
    with open(OUTPUT_DIR / 'bump_detection.pkl', 'rb') as f:
        return pickle.load(f)

def detect_road_anomalies(frame, prev_frame=None):
    """detect potential bump-causing features like cracks, edges, texture changes"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    #focus on bottom 2/3 of frame (road ahead)
    roi_start = h // 3
    roi = gray[roi_start:, :]
    
    detections = []
    
    #1. edge detection for cracks and lines
    edges = cv2.Canny(roi, 50, 150)
    
    #2. find strong horizontal lines (typical of bumps/cracks)
    kernel_h = np.ones((1, 15), np.uint8)
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
    
    #3. find connected components (potential crack regions)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    
    #find contours of edge regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  #skip tiny regions
            continue
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        #offset y to account for ROI
        y_abs = y + roi_start
        
        #calculate feature intensity in this region
        region = roi[y:y+ch, x:x+cw]
        edge_region = edges[y:y+ch, x:x+cw]
        edge_density = edge_region.mean() / 255
        
        #check if it's likely a crack/bump feature
        aspect_ratio = cw / (ch + 1e-6)
        
        #horizontal features are more likely bumps
        is_horizontal = aspect_ratio > 2
        
        #texture analysis
        laplacian = cv2.Laplacian(region, cv2.CV_64F)
        texture_var = laplacian.var()
        
        #score this detection
        score = edge_density * 0.4 + (1 if is_horizontal else 0.3) * 0.3 + min(texture_var / 1000, 1) * 0.3
        
        if score > 0.15 and area > 100:
            detections.append({
                'bbox': (x, y_abs, cw, ch),
                'score': score,
                'edge_density': edge_density,
                'is_horizontal': is_horizontal,
                'area': area,
                'type': 'horizontal_line' if is_horizontal else 'texture_anomaly'
            })
    
    #sort by score
    detections.sort(key=lambda x: x['score'], reverse=True)
    
    #keep top detections
    return detections[:5]

def detect_texture_change(frame, prev_frame):
    """detect sudden texture changes between frames"""
    if prev_frame is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(float)
    
    h, w = gray.shape
    roi_start = h // 3
    
    #compute difference
    diff = np.abs(gray[roi_start:, :] - prev_gray[roi_start:, :])
    
    #threshold significant changes
    threshold = diff.mean() + 2 * diff.std()
    significant = (diff > threshold).astype(np.uint8) * 255
    
    #find regions of change
    dilated = cv2.dilate(significant, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    changes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            x, y, cw, ch = cv2.boundingRect(cnt)
            changes.append({
                'bbox': (x, y + roi_start, cw, ch),
                'area': area,
                'type': 'texture_change'
            })
    
    return changes[:3]

def draw_detections(frame, detections, bump_info=None):
    """draw detection boxes and labels on frame"""
    frame_out = frame.copy()
    h, w = frame.shape[:2]
    
    #color scheme
    colors = {
        'horizontal_line': (0, 255, 255),    #yellow - likely bump/crack
        'texture_anomaly': (255, 165, 0),    #orange - texture issue
        'texture_change': (255, 0, 255),     #magenta - motion/change
    }
    
    for det in detections:
        x, y, cw, ch = det['bbox']
        det_type = det.get('type', 'unknown')
        color = colors.get(det_type, (255, 255, 255))
        
        #draw bounding box
        cv2.rectangle(frame_out, (x, y), (x + cw, y + ch), color, 2)
        
        #label
        label = det_type.replace('_', ' ')
        if 'score' in det:
            label += f" ({det['score']:.2f})"
        
        #put label above box
        label_y = max(y - 5, 15)
        cv2.putText(frame_out, label, (x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    #add bump warning if applicable
    if bump_info:
        frames_until, intensity = bump_info
        
        #warning bar at top
        warning_height = 25
        alpha = min(1.0, intensity / 10)  #fade based on intensity
        
        #red gradient warning
        overlay = frame_out.copy()
        cv2.rectangle(overlay, (0, 0), (w, warning_height), (0, 0, 255), -1)
        frame_out = cv2.addWeighted(overlay, alpha * 0.7, frame_out, 1 - alpha * 0.7, 0)
        
        #warning text
        warning_text = f"BUMP IN {frames_until} FRAMES!"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame_out, warning_text, (text_x, 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        #pulsing border
        border_width = max(2, int(5 * alpha))
        cv2.rectangle(frame_out, (0, 0), (w-1, h-1), (0, 0, 255), border_width)
    
    return frame_out

def generate_annotated_video(video_path, bump_frames, bump_intensities, output_path):
    """generate video with bump-causing objects labeled"""
    print(f"generating annotated video...")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    #create bump lookup for quick access
    bump_dict = {bf: bi for bf, bi in zip(bump_frames, bump_intensities)}
    
    #find frames that are approaching a bump
    warning_frames = set()
    for bf in bump_frames:
        for i in range(max(0, bf - LOOKAHEAD_FRAMES), bf):
            warning_frames.add(i)
    
    prev_frame = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #detect anomalies in current frame
        detections = detect_road_anomalies(frame, prev_frame)
        
        #add texture change detection
        tex_changes = detect_texture_change(frame, prev_frame)
        detections.extend(tex_changes)
        
        #check if we're approaching a bump
        bump_info = None
        if frame_idx in warning_frames:
            #find the upcoming bump
            for bf in bump_frames:
                if bf > frame_idx and bf <= frame_idx + LOOKAHEAD_FRAMES:
                    frames_until = bf - frame_idx
                    intensity = bump_dict.get(bf, 5)
                    bump_info = (frames_until, intensity)
                    break
        
        #check if this is a bump frame
        if frame_idx in bump_dict:
            #highlight this is the bump moment
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            cv2.putText(frame, "BUMP!", (width//2 - 30, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        #draw annotations
        frame_out = draw_detections(frame, detections, bump_info)
        
        #add frame counter
        cv2.putText(frame_out, f"Frame: {frame_idx}", (5, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        out.write(frame_out)
        prev_frame = frame
        frame_idx += 1
        
        if frame_idx % 500 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
    
    cap.release()
    out.release()
    print(f"  saved to {output_path}")
    return frame_idx

def generate_bump_clips(video_path, bump_frames, bump_intensities, output_dir, clip_before=30, clip_after=15):
    """generate individual clips around each bump"""
    print(f"\ngenerating bump clips...")
    
    clips_dir = output_dir / "bump_clips"
    clips_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #read all frames (for small scaled video this is okay)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    #generate clip for top bumps
    bump_data = list(zip(bump_frames, bump_intensities))
    bump_data.sort(key=lambda x: x[1], reverse=True)  #sort by intensity
    
    n_clips = min(20, len(bump_data))  #top 20 bumps
    
    for i, (bf, intensity) in enumerate(bump_data[:n_clips]):
        start = max(0, bf - clip_before)
        end = min(len(frames), bf + clip_after)
        
        clip_path = clips_dir / f"bump_{i+1:02d}_frame{bf}_intensity{intensity:.1f}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        
        for j, frame_idx in enumerate(range(start, end)):
            frame = frames[frame_idx].copy()
            
            #detect anomalies
            prev_frame = frames[frame_idx - 1] if frame_idx > 0 else None
            detections = detect_road_anomalies(frame, prev_frame)
            
            #calculate frames until bump
            frames_until = bf - frame_idx
            
            if frames_until > 0:
                bump_info = (frames_until, intensity)
            else:
                bump_info = None
            
            #mark bump frame
            if frame_idx == bf:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                cv2.putText(frame, "BUMP!", (width//2 - 30, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            frame_out = draw_detections(frame, detections, bump_info)
            
            #add clip info
            cv2.putText(frame_out, f"Bump #{i+1} | Intensity: {intensity:.1f}", 
                       (5, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            cv2.putText(frame_out, f"Frame: {frame_idx} | {'BUMP' if frame_idx == bf else f'{frames_until} frames to bump' if frames_until > 0 else f'{-frames_until} after'}", 
                       (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            out.write(frame_out)
        
        out.release()
        print(f"  clip {i+1}/{n_clips}: bump at frame {bf}, intensity {intensity:.1f}")
    
    print(f"  saved {n_clips} clips to {clips_dir}")

def main():
    #find video
    video_files = list(DATA_DIR.glob("*.mp4")) + list(DATA_DIR.glob("*.mov"))
    if not video_files:
        print("no videos found!")
        return
    
    video_path = video_files[0]
    print(f"video: {video_path.name}")
    
    #load bump detection results
    print("loading bump detection results...")
    detection = load_bump_detection()
    bump_frames = detection['bump_frames']
    bump_intensities = detection['bump_intensities']
    print(f"  {len(bump_frames)} bumps detected")
    
    #generate full annotated video
    output_path = OUTPUT_DIR / "bump_annotated_video.mp4"
    generate_annotated_video(video_path, bump_frames, bump_intensities, output_path)
    
    #generate individual bump clips
    generate_bump_clips(video_path, bump_frames, bump_intensities, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\noutput files:")
    print(f"  - {OUTPUT_DIR / 'bump_annotated_video.mp4'} (full video)")
    print(f"  - {OUTPUT_DIR / 'bump_clips/'} (individual bump clips)")

if __name__ == "__main__":
    main()




