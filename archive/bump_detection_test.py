#test script - short version with no display
import matplotlib
matplotlib.use('Agg')  #non-interactive backend - fixes tkinter error

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

#config
DATA_DIR = Path("data-scaled")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
FPS = 20
MAX_FRAMES = 2000  #limit for testing

print(f"=== SHORT TEST (first {MAX_FRAMES} frames) ===\n")

#load video
def load_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames)

def compute_optical_flow(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    flows = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        if (i + 1) % 200 == 0:
            print(f"optical flow: {i+1}/{len(grays)-1}")
    return flows

class RoadHazardDetector:
    def __init__(self, roi_top_ratio=0.4, roi_bottom_ratio=0.95):
        self.roi_top = roi_top_ratio
        self.roi_bottom = roi_bottom_ratio
        
    def detect_hazards(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape
        roi_y1 = int(h * self.roi_top)
        roi_y2 = int(h * self.roi_bottom)
        roi = gray[roi_y1:roi_y2, :]
        roi_h = roi_y2 - roi_y1
        
        edges = cv2.Canny(roi, 30, 100)
        blur = cv2.GaussianBlur(roi, (21, 21), 0)
        local_diff = cv2.absdiff(roi, blur)
        _, anomaly_mask = cv2.threshold(local_diff, 15, 255, cv2.THRESH_BINARY)
        lap = cv2.Laplacian(roi, cv2.CV_64F)
        lap_abs = np.abs(lap).astype(np.uint8)
        _, rough_mask = cv2.threshold(lap_abs, 20, 255, cv2.THRESH_BINARY)
        
        combined = cv2.bitwise_or(edges, anomaly_mask)
        combined = cv2.bitwise_or(combined, rough_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hazards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > w * roi_h * 0.3:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = x + bw // 2, y + bh // 2
            hazard_roi = roi[max(0,y-5):min(roi_h,y+bh+5), max(0,x-5):min(w,x+bw+5)]
            if hazard_roi.size == 0:
                continue
            hazard = {
                'x': cx, 'y': cy + roi_y1,
                'width': bw, 'height': bh, 'area': area,
                'intensity_mean': hazard_roi.mean(),
                'intensity_std': hazard_roi.std(),
                'edge_density': edges[y:y+bh, x:x+bw].mean() / 255 if bh > 0 and bw > 0 else 0,
                'y_normalized': (cy + roi_y1) / h,
                'x_normalized': cx / w,
            }
            hazards.append(hazard)
        return hazards, combined, roi_y1

class HazardTracker:
    def __init__(self, max_age=15, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.track_id = 0
        self.completed = []
        
    def update(self, hazards, frame_idx):
        for t in self.tracks:
            if t['positions'] and len(t['positions']) >= 2:
                dy = t['positions'][-1]['y'] - t['positions'][-2]['y']
                dx = t['positions'][-1]['x'] - t['positions'][-2]['x']
                t['pred_y'] = t['positions'][-1]['y'] + dy
                t['pred_x'] = t['positions'][-1]['x'] + dx
                t['vel_y'], t['vel_x'] = dy, dx
            elif t['positions']:
                t['pred_y'] = t['positions'][-1]['y'] + 2
                t['pred_x'] = t['positions'][-1]['x']
                t['vel_y'], t['vel_x'] = 2, 0
        
        matched_t, matched_h = set(), set()
        for i, h in enumerate(hazards):
            best_j, best_d = None, float('inf')
            for j, t in enumerate(self.tracks):
                if j in matched_t or 'pred_y' not in t:
                    continue
                d = np.sqrt((h['x']-t['pred_x'])**2 + (h['y']-t['pred_y'])**2)
                if t['positions'] and h['y'] < t['positions'][-1]['y'] - 5:
                    continue
                if d < 50 and d < best_d:
                    best_d, best_j = d, j
            if best_j is not None:
                matched_t.add(best_j)
                matched_h.add(i)
                t = self.tracks[best_j]
                t['positions'].append({'x': h['x'], 'y': h['y'], 'frame': frame_idx, 'features': h})
                t['hits'] += 1
                t['age'] = 0
                t['last_seen'] = frame_idx
        
        for i, h in enumerate(hazards):
            if i not in matched_h:
                self.tracks.append({
                    'id': self.track_id, 
                    'positions': [{'x': h['x'], 'y': h['y'], 'frame': frame_idx, 'features': h}],
                    'hits': 1, 'age': 0, 'start_frame': frame_idx, 'last_seen': frame_idx,
                    'vel_y': 0, 'vel_x': 0, 'caused_bump': False, 'bump_frame': None
                })
                self.track_id += 1
        
        new_tracks = []
        for j, t in enumerate(self.tracks):
            if j not in matched_t:
                t['age'] += 1
            if t['age'] < self.max_age:
                new_tracks.append(t)
            elif t['hits'] >= self.min_hits:
                self.completed.append(t)
        self.tracks = new_tracks
        return [t for t in self.tracks if t['hits'] >= self.min_hits]
    
    def get_all_tracks(self):
        return self.tracks + self.completed
    
    def label_bumps(self, bump_frames, lookahead=10):
        for t in self.get_all_tracks():
            if not t['positions']:
                continue
            last_f = t['positions'][-1]['frame']
            for bf in bump_frames:
                if last_f <= bf <= last_f + lookahead:
                    t['caused_bump'] = True
                    t['bump_frame'] = bf
                    break
        return self.get_all_tracks()

#load video
video_files = list(DATA_DIR.glob("*.mp4"))
if not video_files:
    print("no videos found!")
    exit()

video_path = video_files[0]
print(f"loading {video_path.name} (first {MAX_FRAMES} frames)...")
frames = load_video(video_path, MAX_FRAMES)
print(f"loaded {len(frames)} frames, shape: {frames[0].shape}")
frame_h, frame_w = frames[0].shape[:2]

#compute optical flow
print("\ncomputing optical flow...")
flows = compute_optical_flow(frames)
print(f"computed {len(flows)} flow fields")

#extract flow features for bump detection
def extract_flow_features(flows):
    features = {'avg_vy': [], 'std_vy': [], 'magnitude': []}
    for flow in flows:
        vy = flow[..., 1]
        vx = flow[..., 0]
        features['avg_vy'].append(vy.mean())
        features['std_vy'].append(vy.std())
        features['magnitude'].append(np.sqrt(vx**2 + vy**2).mean())
    return {k: np.array(v) for k, v in features.items()}

print("\nextracting flow features...")
flow_features = extract_flow_features(flows)

#detect bumps (ground truth)
def detect_bumps(flow_features, threshold_pct=95, min_dist=5):
    avg_vy = flow_features['avg_vy']
    vy_diff = np.abs(np.diff(avg_vy))
    vy_abs = np.abs(avg_vy[1:])
    bump_signal = vy_diff + 0.5 * vy_abs
    bump_signal = gaussian_filter1d(bump_signal, sigma=1)
    threshold = np.percentile(bump_signal, threshold_pct)
    peaks, props = signal.find_peaks(bump_signal, height=threshold, distance=min_dist)
    return peaks + 1, props['peak_heights'], bump_signal, threshold

print("detecting bumps (ground truth)...")
bump_frames, bump_intensities, bump_signal, bump_threshold = detect_bumps(flow_features)
print(f"detected {len(bump_frames)} bumps")

#run hazard tracking
print("\ntracking road hazards...")
detector = RoadHazardDetector(roi_top_ratio=0.35, roi_bottom_ratio=0.95)
tracker = HazardTracker(max_age=20, min_hits=3)

for i, frame in enumerate(frames):
    hazards, _, _ = detector.detect_hazards(frame)
    tracker.update(hazards, i)
    if (i + 1) % 500 == 0:
        print(f"frame {i+1}/{len(frames)}, active tracks: {len(tracker.tracks)}")

print(f"\ntotal tracks: {tracker.track_id}")
all_tracks = tracker.get_all_tracks()
confirmed = [t for t in all_tracks if t['hits'] >= 3]
print(f"confirmed tracks (3+ hits): {len(confirmed)}")

#label bump-causing tracks (tighter criteria)
#only tracks that ended near bottom AND within 5 frames of bump
def label_bumps_strict(tracks, bump_frames, frame_h, lookahead=5, min_end_y=0.8):
    """stricter labeling: track must reach bottom AND end near bump"""
    for t in tracks:
        if not t['positions']:
            continue
        last_pos = t['positions'][-1]
        last_f = last_pos['frame']
        last_y_norm = last_pos['y'] / frame_h
        
        #track must reach bottom 80% of frame
        if last_y_norm < min_end_y:
            continue
            
        #check if bump occurred shortly after
        for bf in bump_frames:
            if last_f <= bf <= last_f + lookahead:
                t['caused_bump'] = True
                t['bump_frame'] = bf
                break
    return tracks

labeled_tracks = label_bumps_strict(tracker.get_all_tracks(), bump_frames, frame_h, lookahead=5)
bump_tracks = [t for t in labeled_tracks if t['caused_bump'] and t['hits'] >= 3]
safe_tracks = [t for t in labeled_tracks if not t['caused_bump'] and t['hits'] >= 3]
print(f"bump-causing tracks: {len(bump_tracks)}")
print(f"safe tracks: {len(safe_tracks)}")

#extract track features
def extract_track_features(track, frame_h, frame_w):
    positions = track['positions']
    if len(positions) < 3:
        return None
    features = {}
    features['track_duration'] = positions[-1]['frame'] - positions[0]['frame']
    features['track_hits'] = track['hits']
    y_vals = [p['y'] for p in positions]
    x_vals = [p['x'] for p in positions]
    features['total_dy'] = y_vals[-1] - y_vals[0]
    features['total_dx'] = abs(x_vals[-1] - x_vals[0])
    features['avg_vel_y'] = features['total_dy'] / max(1, len(positions) - 1)
    features['avg_vel_x'] = features['total_dx'] / max(1, len(positions) - 1)
    if len(positions) >= 3:
        vel_y = [y_vals[i+1] - y_vals[i] for i in range(len(y_vals)-1)]
        features['vel_y_std'] = np.std(vel_y)
        features['vel_y_max'] = max(vel_y)
    else:
        features['vel_y_std'] = 0
        features['vel_y_max'] = features['avg_vel_y']
    features['start_y_norm'] = positions[0]['y'] / frame_h
    features['end_y_norm'] = positions[-1]['y'] / frame_h
    features['start_x_norm'] = positions[0]['x'] / frame_w
    features['end_x_norm'] = positions[-1]['x'] / frame_w
    center_dist = [abs(p['x'] / frame_w - 0.5) for p in positions]
    features['min_center_dist'] = min(center_dist)
    features['avg_center_dist'] = np.mean(center_dist)
    first_f = positions[0]['features']
    last_f = positions[-1]['features']
    features['first_area'] = first_f['area']
    features['last_area'] = last_f['area']
    features['area_growth'] = last_f['area'] / (first_f['area'] + 1)
    features['first_intensity'] = first_f['intensity_mean']
    features['last_intensity'] = last_f['intensity_mean']
    features['intensity_change'] = last_f['intensity_mean'] - first_f['intensity_mean']
    features['first_edge_density'] = first_f['edge_density']
    features['last_edge_density'] = last_f['edge_density']
    areas = [p['features']['area'] for p in positions]
    intensities = [p['features']['intensity_mean'] for p in positions]
    edges = [p['features']['edge_density'] for p in positions]
    features['max_area'] = max(areas)
    features['avg_intensity'] = np.mean(intensities)
    features['intensity_std'] = np.std(intensities)
    features['max_edge_density'] = max(edges)
    features['avg_edge_density'] = np.mean(edges)
    return features

#create training data
print("\ncreating training data...")
X_tracks = []
y_tracks = []
for track in labeled_tracks:
    if track['hits'] < 3:
        continue
    features = extract_track_features(track, frame_h, frame_w)
    if features is None or features['total_dy'] < 5:
        continue
    X_tracks.append(list(features.values()))
    y_tracks.append(1 if track['caused_bump'] else 0)

X_tracks = np.array(X_tracks) if X_tracks else np.array([]).reshape(0, 0)
y_tracks = np.array(y_tracks)
feature_names = list(features.keys()) if X_tracks.size > 0 else []

print(f"training samples: {len(X_tracks)}")
print(f"positive (bump-causing): {y_tracks.sum() if len(y_tracks) > 0 else 0}")
print(f"negative (safe): {(1-y_tracks).sum() if len(y_tracks) > 0 else 0}")

#train if we have data
if len(X_tracks) > 5 and y_tracks.sum() > 0:
    print("\ntraining classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tracks, y_tracks, test_size=0.2, random_state=42,
        stratify=y_tracks if y_tracks.sum() >= 2 else None
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    #use balanced class weights to handle imbalance
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {0: cw[0], 1: cw[1]}
    print(f"class weights: safe={cw[0]:.2f}, bump={cw[1]:.2f}")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, 
                                class_weight=cw_dict, random_state=42)
    rf.fit(X_train_s, y_train)
    
    y_pred = rf.predict(X_test_s)
    y_proba = rf.predict_proba(X_test_s)[:, 1]
    
    print("\nclassification report:")
    print(classification_report(y_test, y_pred, target_names=['safe', 'bump'], zero_division=0))
    
    #feature importance
    print("\ntop 10 features:")
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    for i in range(min(10, len(feature_names))):
        print(f"  {feature_names[idx[i]]}: {imp[idx[i]]:.3f}")
    
    #analyze feature differences between classes
    print("\nfeature stats (bump vs safe):")
    bump_mask = y_tracks == 1
    safe_mask = y_tracks == 0
    for i, name in enumerate(feature_names[:8]):  #first 8 features
        bump_mean = X_tracks[bump_mask, i].mean()
        safe_mean = X_tracks[safe_mask, i].mean()
        diff = bump_mean - safe_mean
        print(f"  {name}: bump={bump_mean:.2f}, safe={safe_mean:.2f}, diff={diff:+.2f}")
    
    #save model
    joblib.dump({'model': rf, 'scaler': scaler, 'features': feature_names}, 
                OUTPUT_DIR / 'hazard_predictor_test.joblib')
    print(f"\nmodel saved to {OUTPUT_DIR / 'hazard_predictor_test.joblib'}")
else:
    print("\nnot enough data to train classifier")
    print("(this is expected with only 2000 frames - may not have enough bump events)")

#summary
print("\n" + "="*50)
print("TEST COMPLETE")
print("="*50)
print(f"frames processed: {len(frames)}")
print(f"bumps detected: {len(bump_frames)}")
print(f"tracks created: {tracker.track_id}")
print(f"bump-causing tracks: {len(bump_tracks)}")
print(f"safe tracks: {len(safe_tracks)}")

