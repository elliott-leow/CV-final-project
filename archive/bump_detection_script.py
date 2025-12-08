#imports and setup
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

#ml imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import joblib


#config
DATA_DIR = Path("data-scaled")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS = 20  #target fps after scaling
SEGMENT_LENGTH = 10  #frames per segment
BUMP_THRESHOLD_PERCENTILE = 95  #percentile for bump detection
LOOKAHEAD_FRAMES = 10  #how many frames ahead to look for prediction


def load_video(video_path):
    """load video frames as numpy array"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

def compute_optical_flow(frames):
    """compute dense optical flow between consecutive frames"""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    flows = []
    
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(flow)
        
        if (i + 1) % 100 == 0:
            print(f"processed {i + 1}/{len(grays) - 1} frames")
    
    return flows


#road hazard detection and tracking classes
class RoadHazardDetector:
    """detects potential bump-causing features on road surface"""
    
    def __init__(self, roi_top_ratio=0.4, roi_bottom_ratio=0.95):
        self.roi_top = roi_top_ratio
        self.roi_bottom = roi_bottom_ratio
        
    def detect_hazards(self, frame):
        """detect potential road hazards (cracks, holes, bumps, debris)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape
        
        roi_y1 = int(h * self.roi_top)
        roi_y2 = int(h * self.roi_bottom)
        roi = gray[roi_y1:roi_y2, :]
        roi_h = roi_y2 - roi_y1
        
        hazards = []
        
        #edge-based detection (cracks, edges)
        edges = cv2.Canny(roi, 30, 100)
        
        #local contrast anomalies (dark spots = holes)
        blur = cv2.GaussianBlur(roi, (21, 21), 0)
        local_diff = cv2.absdiff(roi, blur)
        _, anomaly_mask = cv2.threshold(local_diff, 15, 255, cv2.THRESH_BINARY)
        
        #texture roughness via laplacian
        lap = cv2.Laplacian(roi, cv2.CV_64F)
        lap_abs = np.abs(lap).astype(np.uint8)
        _, rough_mask = cv2.threshold(lap_abs, 20, 255, cv2.THRESH_BINARY)
        
        #combine and cleanup
        combined = cv2.bitwise_or(edges, anomaly_mask)
        combined = cv2.bitwise_or(combined, rough_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
    """tracks hazards as they approach (move down in frame)"""
    
    def __init__(self, max_age=15, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.track_id = 0
        self.completed = []
        
    def update(self, hazards, frame_idx):
        #predict positions (hazards move DOWN in frame)
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
                    'id': self.track_id, 'positions': [{'x': h['x'], 'y': h['y'], 'frame': frame_idx, 'features': h}],
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

print("hazard detection classes defined")


#load videos from data-scaled
video_files = list(DATA_DIR.glob("*.mp4")) + list(DATA_DIR.glob("*.mov"))
print(f"found {len(video_files)} video(s): {[v.name for v in video_files]}")

#load first video for processing
if video_files:
    video_path = video_files[0]
    print(f"\nloading {video_path.name}...")
    frames = load_video(video_path)
    print(f"loaded {len(frames)} frames, shape: {frames[0].shape}")
else:
    print("no videos found! run scale_videos.py first.")


#compute optical flow
print("computing optical flow...")
flows = compute_optical_flow(frames)
print(f"computed {len(flows)} flow fields")


#run hazard detection and tracking on all frames
print("detecting and tracking road hazards...")
hazard_detector = RoadHazardDetector(roi_top_ratio=0.35, roi_bottom_ratio=0.95)
hazard_tracker = HazardTracker(max_age=20, min_hits=3)

all_hazards = []
all_active_tracks = []
frame_h, frame_w = frames[0].shape[:2]

for i, frame in enumerate(frames):
    hazards, mask, roi_y1 = hazard_detector.detect_hazards(frame)
    active = hazard_tracker.update(hazards, i)
    
    all_hazards.append(hazards)
    all_active_tracks.append([dict(t) for t in active])
    
    if (i + 1) % 100 == 0:
        approaching = len([t for t in active if t.get('vel_y', 0) > 1])
        print(f"frame {i+1}/{len(frames)}, hazards: {len(hazards)}, tracks: {len(active)}, approaching: {approaching}")

print(f"\ntotal tracks created: {hazard_tracker.track_id}")
print(f"confirmed tracks (3+ hits): {len([t for t in hazard_tracker.get_all_tracks() if t['hits'] >= 3])}")


def extract_flow_features(flows):
    """extract vertical flow statistics per frame"""
    features = {
        'avg_vy': [],      #average vertical flow
        'std_vy': [],      #std of vertical flow
        'max_vy': [],      #max vertical flow
        'min_vy': [],      #min vertical flow
        'avg_vx': [],      #average horizontal flow
        'magnitude': [],   #average flow magnitude
        'bottom_vy': [],   #vertical flow in bottom half (closer to scooter)
    }
    
    for flow in flows:
        vx, vy = flow[..., 0], flow[..., 1]
        h = flow.shape[0]
        
        features['avg_vy'].append(vy.mean())
        features['std_vy'].append(vy.std())
        features['max_vy'].append(vy.max())
        features['min_vy'].append(vy.min())
        features['avg_vx'].append(vx.mean())
        features['magnitude'].append(np.sqrt(vx**2 + vy**2).mean())
        features['bottom_vy'].append(vy[h//2:, :].mean())  #bottom half
    
    return {k: np.array(v) for k, v in features.items()}

flow_features = extract_flow_features(flows)
print(f"extracted features for {len(flow_features['avg_vy'])} frames")


#plot vertical optical flow over time
time = np.arange(len(flow_features['avg_vy'])) / FPS

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

axes[0].plot(time, flow_features['avg_vy'], 'b-', alpha=0.7, label='avg vertical flow')
axes[0].set_ylabel('avg vy (pixels/frame)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, flow_features['std_vy'], 'r-', alpha=0.7, label='std vertical flow')
axes[1].set_ylabel('std vy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, flow_features['magnitude'], 'g-', alpha=0.7, label='flow magnitude')
axes[2].set_ylabel('magnitude')
axes[2].set_xlabel('time (s)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('optical flow analysis over time')
plt.tight_layout()
plt.show()


def detect_bumps(flow_features, threshold_percentile=95, min_distance=5):
    """detect bumps from sudden changes in vertical flow"""
    avg_vy = flow_features['avg_vy']
    
    #compute derivative (change in vertical flow)
    vy_diff = np.abs(np.diff(avg_vy))
    
    #also consider absolute flow magnitude as bump indicator
    vy_abs = np.abs(avg_vy[1:])  #align with diff
    
    #combine signals: sudden change + high magnitude
    bump_signal = vy_diff + 0.5 * vy_abs
    
    #smooth slightly to reduce noise
    bump_signal_smooth = gaussian_filter1d(bump_signal, sigma=1)
    
    #threshold based on percentile
    threshold = np.percentile(bump_signal_smooth, threshold_percentile)
    
    #find peaks above threshold
    peaks, properties = signal.find_peaks(
        bump_signal_smooth, 
        height=threshold,
        distance=min_distance
    )
    
    #offset by 1 since diff reduces length
    bump_frames = peaks + 1
    bump_intensities = properties['peak_heights']
    
    return bump_frames, bump_intensities, bump_signal_smooth, threshold

bump_frames, bump_intensities, bump_signal, bump_threshold = detect_bumps(
    flow_features, 
    threshold_percentile=BUMP_THRESHOLD_PERCENTILE
)

print(f"detected {len(bump_frames)} bumps")
print(f"bump threshold: {bump_threshold:.3f}")
print(f"bump frames: {bump_frames[:20]}..." if len(bump_frames) > 20 else f"bump frames: {bump_frames}")


#visualize bump detection
time_signal = np.arange(len(bump_signal)) / FPS

plt.figure(figsize=(14, 5))
plt.plot(time_signal, bump_signal, 'b-', alpha=0.7, label='bump signal')
plt.axhline(y=bump_threshold, color='r', linestyle='--', label=f'threshold ({BUMP_THRESHOLD_PERCENTILE}th percentile)')

#mark detected bumps
bump_times = (bump_frames - 1) / FPS  #offset to align with signal
plt.scatter(bump_times, bump_intensities, c='red', s=100, marker='v', label='detected bumps', zorder=5)

plt.xlabel('time (s)')
plt.ylabel('bump signal intensity')
plt.title('bump detection from vertical optical flow')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#create frame-level labels (1 = bump, 0 = no bump)
num_frames = len(frames)
bump_labels = np.zeros(num_frames, dtype=int)

#mark bump frames and nearby frames (±2 frames window)
for bf in bump_frames:
    start = max(0, bf - 2)
    end = min(num_frames, bf + 3)
    bump_labels[start:end] = 1

print(f"total frames: {num_frames}")
print(f"bump frames: {bump_labels.sum()} ({100*bump_labels.mean():.1f}%)")
print(f"non-bump frames: {num_frames - bump_labels.sum()}")


#label tracks that caused bumps and extract features
print("labeling bump-causing tracks...")
labeled_tracks = hazard_tracker.label_bumps(bump_frames, lookahead=15)

#count stats
bump_tracks = [t for t in labeled_tracks if t['caused_bump'] and t['hits'] >= 3]
non_bump_tracks = [t for t in labeled_tracks if not t['caused_bump'] and t['hits'] >= 3]
print(f"bump-causing tracks: {len(bump_tracks)}")
print(f"non-bump tracks: {len(non_bump_tracks)}")

def extract_track_features(track):
    """extract features from a tracked hazard for classification"""
    positions = track['positions']
    if len(positions) < 3:
        return None
    
    features = {}
    
    #track duration and persistence
    features['track_duration'] = positions[-1]['frame'] - positions[0]['frame']
    features['track_hits'] = track['hits']
    
    #trajectory features (how it approached)
    y_vals = [p['y'] for p in positions]
    x_vals = [p['x'] for p in positions]
    
    #total vertical travel (should be positive = moving down/approaching)
    features['total_dy'] = y_vals[-1] - y_vals[0]
    features['total_dx'] = abs(x_vals[-1] - x_vals[0])
    
    #average velocity
    features['avg_vel_y'] = features['total_dy'] / (len(positions) - 1) if len(positions) > 1 else 0
    features['avg_vel_x'] = features['total_dx'] / (len(positions) - 1) if len(positions) > 1 else 0
    
    #velocity consistency (low std = consistent approach)
    if len(positions) >= 3:
        vel_y = [y_vals[i+1] - y_vals[i] for i in range(len(y_vals)-1)]
        features['vel_y_std'] = np.std(vel_y)
        features['vel_y_max'] = max(vel_y)
    else:
        features['vel_y_std'] = 0
        features['vel_y_max'] = features['avg_vel_y']
    
    #where hazard started and ended (normalized 0-1)
    features['start_y_norm'] = positions[0]['y'] / frame_h
    features['end_y_norm'] = positions[-1]['y'] / frame_h
    features['start_x_norm'] = positions[0]['x'] / frame_w
    features['end_x_norm'] = positions[-1]['x'] / frame_w
    
    #hazard is in the center path? (x near 0.5 is more dangerous)
    center_dist = [abs(p['x'] / frame_w - 0.5) for p in positions]
    features['min_center_dist'] = min(center_dist)
    features['avg_center_dist'] = np.mean(center_dist)
    
    #visual features from first and last detection
    first_f = positions[0]['features']
    last_f = positions[-1]['features']
    
    features['first_area'] = first_f['area']
    features['last_area'] = last_f['area']
    features['area_growth'] = last_f['area'] / (first_f['area'] + 1)  #grows as approaches
    
    features['first_intensity'] = first_f['intensity_mean']
    features['last_intensity'] = last_f['intensity_mean']
    features['intensity_change'] = last_f['intensity_mean'] - first_f['intensity_mean']
    
    features['first_edge_density'] = first_f['edge_density']
    features['last_edge_density'] = last_f['edge_density']
    
    #aggregate features across track
    areas = [p['features']['area'] for p in positions]
    intensities = [p['features']['intensity_mean'] for p in positions]
    edges = [p['features']['edge_density'] for p in positions]
    
    features['max_area'] = max(areas)
    features['avg_intensity'] = np.mean(intensities)
    features['intensity_std'] = np.std(intensities)
    features['max_edge_density'] = max(edges)
    features['avg_edge_density'] = np.mean(edges)
    
    return features

#create training data from tracks
print("\ncreating training data from tracked hazards...")
X_tracks = []
y_tracks = []
track_ids = []

for track in labeled_tracks:
    if track['hits'] < 3:
        continue
    features = extract_track_features(track)
    if features is None:
        continue
    
    #only include tracks that actually approached (moved downward)
    if features['total_dy'] < 5:  #min 5 pixels of downward movement
        continue
    
    X_tracks.append(list(features.values()))
    y_tracks.append(1 if track['caused_bump'] else 0)
    track_ids.append(track['id'])

X_tracks = np.array(X_tracks)
y_tracks = np.array(y_tracks)
feature_names_tracks = list(features.keys())

print(f"\ntrack-based training data: X={X_tracks.shape}, y={y_tracks.shape}")
print(f"positive (bump-causing): {y_tracks.sum()} ({100*y_tracks.mean():.1f}%)")
print(f"negative (non-bump): {(1-y_tracks).sum()}")
print(f"features: {len(feature_names_tracks)}")


#train classifier on track features
print("training hazard bump classifier...")

if len(X_tracks) > 10 and y_tracks.sum() > 0:
    #split data
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_tracks, y_tracks, test_size=0.2, random_state=42, 
        stratify=y_tracks if y_tracks.sum() >= 2 else None
    )
    
    #scale features
    scaler_tracks = StandardScaler()
    X_train_ts = scaler_tracks.fit_transform(X_train_t)
    X_test_ts = scaler_tracks.transform(X_test_t)
    
    #class weights for imbalance
    if len(np.unique(y_train_t)) > 1:
        cw = compute_class_weight('balanced', classes=np.unique(y_train_t), y=y_train_t)
        cw_dict = {i: w for i, w in enumerate(cw)}
    else:
        cw_dict = {0: 1.0, 1: 1.0}
    print(f"class weights: {cw_dict}")
    
    #train random forest on track features
    rf_tracks = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=3,
        class_weight=cw_dict, random_state=42, n_jobs=-1
    )
    rf_tracks.fit(X_train_ts, y_train_t)
    
    #evaluate
    y_pred_t = rf_tracks.predict(X_test_ts)
    print("\ntrack classifier evaluation:")
    print(classification_report(y_test_t, y_pred_t, target_names=['no bump', 'bump'], zero_division=0))
    
    #feature importance
    importances = rf_tracks.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\ntop 10 features for bump prediction:")
    for i in range(min(10, len(feature_names_tracks))):
        print(f"  {feature_names_tracks[indices[i]]}: {importances[indices[i]]:.4f}")
else:
    print("not enough training data for track classifier")
    rf_tracks = None
    scaler_tracks = None


class ApproachingHazardPredictor:
    """real-time bump prediction based on approaching hazards"""
    
    def __init__(self, model, scaler, feature_names, frame_shape):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.frame_h, self.frame_w = frame_shape[:2]
        
        self.detector = RoadHazardDetector(roi_top_ratio=0.35, roi_bottom_ratio=0.95)
        self.tracker = HazardTracker(max_age=20, min_hits=3)
        self.frame_idx = 0
        
        self.alert_active = False
        self.alert_probability = 0.0
        self.dangerous_tracks = []
        
    def process_frame(self, frame):
        """process frame and predict if approaching hazard will cause bump"""
        hazards, mask, roi_y1 = self.detector.detect_hazards(frame)
        active_tracks = self.tracker.update(hazards, self.frame_idx)
        
        #check each active approaching track
        max_prob = 0.0
        dangerous = []
        
        for track in active_tracks:
            #only evaluate tracks that are actively approaching
            if track.get('vel_y', 0) < 1 or len(track['positions']) < 3:
                continue
                
            #extract features and predict
            features = self._extract_live_features(track)
            if features is None:
                continue
            
            X = np.array([list(features.values())])
            X_scaled = self.scaler.transform(X)
            prob = self.model.predict_proba(X_scaled)[0, 1]
            
            if prob > 0.3:  #moderate confidence threshold
                dangerous.append({
                    'track': track,
                    'probability': prob,
                    'position': (track['positions'][-1]['x'], track['positions'][-1]['y'])
                })
                max_prob = max(max_prob, prob)
        
        self.dangerous_tracks = dangerous
        self.alert_probability = max_prob
        self.alert_active = max_prob > 0.5
        self.frame_idx += 1
        
        return self.alert_active, self.alert_probability, dangerous
    
    def _extract_live_features(self, track):
        """extract features from live track (same as training)"""
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
        features['avg_vel_y'] = features['total_dy'] / (len(positions) - 1) if len(positions) > 1 else 0
        features['avg_vel_x'] = features['total_dx'] / (len(positions) - 1) if len(positions) > 1 else 0
        
        if len(positions) >= 3:
            vel_y = [y_vals[i+1] - y_vals[i] for i in range(len(y_vals)-1)]
            features['vel_y_std'] = np.std(vel_y)
            features['vel_y_max'] = max(vel_y)
        else:
            features['vel_y_std'] = 0
            features['vel_y_max'] = features['avg_vel_y']
        
        features['start_y_norm'] = positions[0]['y'] / self.frame_h
        features['end_y_norm'] = positions[-1]['y'] / self.frame_h
        features['start_x_norm'] = positions[0]['x'] / self.frame_w
        features['end_x_norm'] = positions[-1]['x'] / self.frame_w
        
        center_dist = [abs(p['x'] / self.frame_w - 0.5) for p in positions]
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
    
    def reset(self):
        self.tracker = HazardTracker(max_age=20, min_hits=3)
        self.frame_idx = 0
        self.alert_active = False
        self.alert_probability = 0.0
        self.dangerous_tracks = []

print("ApproachingHazardPredictor class defined")


#visualize tracked hazards and bump correlation
def visualize_hazard_tracks(frames, labeled_tracks, bump_frames, n_examples=5):
    """visualize example tracks - bump-causing vs non-bump"""
    bump_tracks = [t for t in labeled_tracks if t['caused_bump'] and t['hits'] >= 5]
    safe_tracks = [t for t in labeled_tracks if not t['caused_bump'] and t['hits'] >= 5]
    
    print(f"bump-causing tracks (5+ hits): {len(bump_tracks)}")
    print(f"safe tracks (5+ hits): {len(safe_tracks)}")
    
    #show examples of each
    for label, tracks in [("BUMP-CAUSING", bump_tracks[:n_examples]), ("SAFE", safe_tracks[:n_examples])]:
        if not tracks:
            continue
        print(f"\n--- {label} HAZARD TRACKS ---")
        
        for i, track in enumerate(tracks):
            positions = track['positions']
            start_frame = positions[0]['frame']
            end_frame = positions[-1]['frame']
            
            #show start, middle, end frames
            mid_frame = start_frame + (end_frame - start_frame) // 2
            show_frames = [start_frame, mid_frame, end_frame]
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            for j, f_idx in enumerate(show_frames):
                if f_idx >= len(frames):
                    continue
                frame = frames[f_idx].copy()
                
                #draw track trajectory up to this point
                track_pts = [(p['x'], p['y']) for p in positions if p['frame'] <= f_idx]
                for k in range(1, len(track_pts)):
                    cv2.line(frame, 
                            (int(track_pts[k-1][0]), int(track_pts[k-1][1])),
                            (int(track_pts[k][0]), int(track_pts[k][1])),
                            (0, 0, 255) if label == "BUMP-CAUSING" else (0, 255, 0), 2)
                
                #mark current position
                if track_pts:
                    cx, cy = int(track_pts[-1][0]), int(track_pts[-1][1])
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255) if label == "BUMP-CAUSING" else (0, 255, 0), -1)
                
                axes[j].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                axes[j].set_title(f"frame {f_idx}")
                axes[j].axis('off')
            
            title = f"{label} track {track['id']}: frames {start_frame}-{end_frame}, hits={track['hits']}"
            if track['caused_bump']:
                title += f", bump@{track['bump_frame']}"
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()

#visualize some example tracks
if labeled_tracks:
    visualize_hazard_tracks(frames, labeled_tracks, bump_frames, n_examples=3)


#run demo: predict bumps from approaching hazards
def run_hazard_prediction_demo(frames, model, scaler, feature_names, output_path=None):
    """run hazard-based bump prediction and optionally save video"""
    if model is None:
        print("no model available for prediction")
        return [], []
    
    predictor = ApproachingHazardPredictor(model, scaler, feature_names, frames[0].shape)
    
    h, w = frames[0].shape[:2]
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, FPS, (w, h))
    
    alerts = []
    probs = []
    
    for i, frame in enumerate(frames):
        alert, prob, dangerous = predictor.process_frame(frame)
        alerts.append(alert)
        probs.append(prob)
        
        #annotate frame
        frame_out = frame.copy()
        
        #draw all active tracks
        for track in predictor.tracker.tracks:
            if track['hits'] >= 3:
                pts = [(p['x'], p['y']) for p in track['positions']]
                for j in range(1, len(pts)):
                    color = (0, 255, 255)  #yellow for tracked
                    cv2.line(frame_out, (int(pts[j-1][0]), int(pts[j-1][1])),
                            (int(pts[j][0]), int(pts[j][1])), color, 2)
        
        #highlight dangerous hazards
        for d in dangerous:
            x, y = int(d['position'][0]), int(d['position'][1])
            p = d['probability']
            cv2.circle(frame_out, (x, y), 15, (0, 0, 255), 3)
            cv2.putText(frame_out, f"{p:.0%}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        #alert overlay
        if alert:
            cv2.rectangle(frame_out, (0, 0), (w-1, h-1), (0, 0, 255), 8)
            cv2.putText(frame_out, f"BUMP AHEAD! ({prob:.0%})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame_out, f"Clear ({prob:.0%})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if out:
            out.write(frame_out)
        
        if (i + 1) % 200 == 0:
            print(f"processed {i+1}/{len(frames)} frames")
    
    if out:
        out.release()
    
    return alerts, probs

print("running hazard-based prediction demo...")
if rf_tracks is not None:
    demo_alerts, demo_probs = run_hazard_prediction_demo(
        frames, rf_tracks, scaler_tracks, feature_names_tracks,
        OUTPUT_DIR / 'hazard_prediction_demo.mp4'
    )
    print(f"\ndemo complete! output: {OUTPUT_DIR / 'hazard_prediction_demo.mp4'}")
    print(f"alerts triggered: {sum(demo_alerts)} frames")
else:
    print("skipping demo - no model trained")


#save hazard prediction model
if rf_tracks is not None:
    model_data = {
        'model': rf_tracks,
        'scaler': scaler_tracks,
        'feature_names': feature_names_tracks,
        'frame_shape': frames[0].shape,
    }
    joblib.dump(model_data, OUTPUT_DIR / 'hazard_predictor.joblib')
    print(f"hazard predictor saved to {OUTPUT_DIR / 'hazard_predictor.joblib'}")

#save track data for analysis
track_data = {
    'labeled_tracks': labeled_tracks,
    'bump_frames': bump_frames,
    'X_tracks': X_tracks,
    'y_tracks': y_tracks,
    'feature_names': feature_names_tracks,
}
with open(OUTPUT_DIR / 'hazard_track_data.pkl', 'wb') as f:
    pickle.dump(track_data, f)
print(f"track data saved to {OUTPUT_DIR / 'hazard_track_data.pkl'}")

#summary
print("\n" + "="*60)
print("HAZARD-BASED BUMP PREDICTION SUMMARY")
print("="*60)
print(f"\nvideo: {video_path.name}")
print(f"frames: {len(frames)}, duration: {len(frames)/FPS:.1f}s")

print(f"\n--- detection ---")
print(f"total hazards tracked: {hazard_tracker.track_id}")
print(f"confirmed tracks (3+ hits): {len([t for t in labeled_tracks if t['hits'] >= 3])}")
print(f"bump-causing tracks: {len([t for t in labeled_tracks if t['caused_bump']])}")
print(f"bumps detected (ground truth): {len(bump_frames)}")

print(f"\n--- prediction model ---")
if rf_tracks is not None:
    print(f"training samples: {len(X_tracks)}")
    print(f"features: {len(feature_names_tracks)}")
    print(f"model: RandomForest")
else:
    print("no model trained (insufficient data)")

print(f"\n--- key features for bump prediction ---")
if rf_tracks is not None:
    for i in range(min(5, len(feature_names_tracks))):
        print(f"  {feature_names_tracks[indices[i]]}: {importances[indices[i]]:.3f}")

print(f"\n--- files saved ---")
print(f"  {OUTPUT_DIR / 'hazard_predictor.joblib'}")
print(f"  {OUTPUT_DIR / 'hazard_track_data.pkl'}")
print(f"  {OUTPUT_DIR / 'hazard_prediction_demo.mp4'}")


def extract_texture_features(frame):
    """extract texture features that might indicate cracks/bumps on sidewalk"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    
    features = {}
    
    #focus on bottom half (closer to scooter, more relevant)
    roi = gray[h//2:, :]
    
    #edge detection (cracks have strong edges)
    edges = cv2.Canny(roi, 50, 150)
    features['edge_density'] = edges.mean() / 255
    features['edge_std'] = edges.std() / 255
    
    #sobel gradients (horizontal lines = bumps/cracks)
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    features['sobel_x_mean'] = np.abs(sobel_x).mean()
    features['sobel_y_mean'] = np.abs(sobel_y).mean()
    features['sobel_ratio'] = features['sobel_y_mean'] / (features['sobel_x_mean'] + 1e-6)
    
    #laplacian (overall texture/roughness)
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    features['laplacian_var'] = laplacian.var()
    features['laplacian_mean'] = np.abs(laplacian).mean()
    
    #histogram features (contrast)
    features['intensity_mean'] = roi.mean()
    features['intensity_std'] = roi.std()
    
    #gabor filter for texture (detects linear patterns like cracks)
    gabor_responses = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0)
        filtered = cv2.filter2D(roi, cv2.CV_64F, kernel)
        gabor_responses.append(np.abs(filtered).mean())
    features['gabor_0'] = gabor_responses[0]
    features['gabor_45'] = gabor_responses[1]
    features['gabor_90'] = gabor_responses[2]
    features['gabor_135'] = gabor_responses[3]
    features['gabor_max'] = max(gabor_responses)
    
    return features

#test on single frame
test_features = extract_texture_features(frames[0])
print("texture features:")
for k, v in test_features.items():
    print(f"  {k}: {v:.4f}")


def extract_segment_features(frames, flows, start_idx, segment_length=10):
    """extract features from a segment of frames for bump prediction"""
    end_idx = min(start_idx + segment_length, len(frames))
    segment_frames = frames[start_idx:end_idx]
    segment_flows = flows[start_idx:min(end_idx, len(flows))]
    
    if len(segment_frames) < segment_length // 2:
        return None
    
    features = {}
    
    #aggregate texture features across segment
    texture_feats = defaultdict(list)
    for frame in segment_frames:
        tf = extract_texture_features(frame)
        for k, v in tf.items():
            texture_feats[k].append(v)
    
    for k, v in texture_feats.items():
        features[f'tex_{k}_mean'] = np.mean(v)
        features[f'tex_{k}_std'] = np.std(v)
        features[f'tex_{k}_max'] = np.max(v)
        features[f'tex_{k}_trend'] = v[-1] - v[0] if len(v) > 1 else 0
    
    #optical flow features in segment
    if len(segment_flows) > 0:
        vy_vals = [flow[..., 1].mean() for flow in segment_flows]
        vx_vals = [flow[..., 0].mean() for flow in segment_flows]
        mag_vals = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean() for flow in segment_flows]
        
        features['flow_vy_mean'] = np.mean(vy_vals)
        features['flow_vy_std'] = np.std(vy_vals)
        features['flow_vy_trend'] = vy_vals[-1] - vy_vals[0] if len(vy_vals) > 1 else 0
        features['flow_vx_mean'] = np.mean(vx_vals)
        features['flow_mag_mean'] = np.mean(mag_vals)
        features['flow_mag_max'] = np.max(mag_vals)
    
    return features

#test
test_seg_features = extract_segment_features(frames, flows, 0, SEGMENT_LENGTH)
print(f"segment features: {len(test_seg_features)} features")


#create training data: segments with labels indicating if bump occurs in next N frames
def create_training_data(frames, flows, bump_frames, segment_length=10, lookahead=10):
    """create training dataset for bump prediction"""
    X = []
    y = []
    segment_indices = []
    
    bump_set = set(bump_frames)
    
    #slide window through video
    step = segment_length // 2  #50% overlap
    
    for start_idx in range(0, len(frames) - segment_length - lookahead, step):
        #extract features from current segment
        features = extract_segment_features(frames, flows, start_idx, segment_length)
        
        if features is None:
            continue
        
        #check if bump occurs in the lookahead window (after segment ends)
        lookahead_start = start_idx + segment_length
        lookahead_end = lookahead_start + lookahead
        has_bump = any(bf in range(lookahead_start, lookahead_end) for bf in bump_set)
        
        X.append(list(features.values()))
        y.append(1 if has_bump else 0)
        segment_indices.append(start_idx)
        
        if len(X) % 50 == 0:
            print(f"processed {len(X)} segments...")
    
    feature_names = list(features.keys()) if features else []
    return np.array(X), np.array(y), segment_indices, feature_names

print("creating training data...")
X, y, segment_indices, feature_names = create_training_data(
    frames, flows, bump_frames, 
    segment_length=SEGMENT_LENGTH, 
    lookahead=LOOKAHEAD_FRAMES
)

print(f"\ntraining data shape: X={X.shape}, y={y.shape}")
print(f"positive samples (bump ahead): {y.sum()} ({100*y.mean():.1f}%)")
print(f"negative samples (no bump): {(1-y).sum()}")
print(f"feature count: {len(feature_names)}")


#handle class imbalance and train model
#split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#compute class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"class weights: {class_weight_dict}")


#train random forest model
print("training random forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

#cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"cross-validation F1 scores: {cv_scores}")
print(f"mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")


#train gradient boosting model
print("training gradient boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

cv_scores_gb = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"GB cross-validation F1 scores: {cv_scores_gb}")
print(f"mean CV F1: {cv_scores_gb.mean():.3f} (+/- {cv_scores_gb.std()*2:.3f})")


#evaluate both models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*50}")
    print(f"{model_name} evaluation:")
    print(f"{'='*50}")
    print("\nclassification report:")
    print(classification_report(y_test, y_pred, target_names=['no bump', 'bump']))
    
    print("confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return y_pred, y_proba

rf_pred, rf_proba = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
gb_pred, gb_proba = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting")


#feature importance analysis
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
top_n = min(20, len(feature_names))
plt.barh(range(top_n), importances[indices[:top_n]][::-1])
plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
plt.xlabel('feature importance')
plt.title('top features for bump prediction')
plt.tight_layout()
plt.show()

print("\ntop 10 most important features:")
for i in range(min(10, len(feature_names))):
    print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


#precision-recall curve
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf_proba)
precision_gb, recall_gb, thresholds_gb = precision_recall_curve(y_test, gb_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall_rf, precision_rf, 'b-', label='random forest')
plt.plot(recall_gb, precision_gb, 'r-', label='gradient boosting')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision-recall curve for bump prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#save best model
best_model = rf_model  #choose based on evaluation

model_data = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'segment_length': SEGMENT_LENGTH,
    'lookahead': LOOKAHEAD_FRAMES
}

joblib.dump(model_data, OUTPUT_DIR / 'bump_predictor.joblib')
print(f"model saved to {OUTPUT_DIR / 'bump_predictor.joblib'}")


#save bump detection results
detection_results = {
    'bump_frames': bump_frames,
    'bump_intensities': bump_intensities,
    'bump_threshold': bump_threshold,
    'bump_labels': bump_labels,
    'flow_features': flow_features,
    'fps': FPS
}

with open(OUTPUT_DIR / 'bump_detection.pkl', 'wb') as f:
    pickle.dump(detection_results, f)
    
print(f"detection results saved to {OUTPUT_DIR / 'bump_detection.pkl'}")


class BumpAlertSystem:
    """real-time bump prediction alert system"""
    
    def __init__(self, model_path):
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.segment_length = data['segment_length']
        self.lookahead = data['lookahead']
        
        #buffer for frames and flows
        self.frame_buffer = []
        self.flow_buffer = []
        self.prev_gray = None
        
        #alert state
        self.alert_active = False
        self.alert_probability = 0.0
        
    def process_frame(self, frame):
        """process new frame and return alert status"""
        self.frame_buffer.append(frame)
        
        #compute optical flow if we have previous frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            self.flow_buffer.append(flow)
        self.prev_gray = gray
        
        #keep buffer at segment length
        if len(self.frame_buffer) > self.segment_length:
            self.frame_buffer.pop(0)
        if len(self.flow_buffer) > self.segment_length:
            self.flow_buffer.pop(0)
        
        #predict if we have enough data
        if len(self.frame_buffer) >= self.segment_length and len(self.flow_buffer) >= self.segment_length - 1:
            features = extract_segment_features(
                np.array(self.frame_buffer), 
                self.flow_buffer, 
                0, 
                self.segment_length
            )
            
            if features:
                X = np.array([list(features.values())])
                X_scaled = self.scaler.transform(X)
                
                prob = self.model.predict_proba(X_scaled)[0, 1]
                self.alert_probability = prob
                self.alert_active = prob > 0.5
        
        return self.alert_active, self.alert_probability
    
    def reset(self):
        """reset buffer state"""
        self.frame_buffer = []
        self.flow_buffer = []
        self.prev_gray = None
        self.alert_active = False
        self.alert_probability = 0.0

print("BumpAlertSystem class defined")


#demo: run alert system on video and visualize
def demo_alert_system(video_path, model_path, output_video_path=None):
    """run bump alert system on video with visualization"""
    alert_system = BumpAlertSystem(model_path)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #setup output video
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    alerts = []
    probabilities = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        alert, prob = alert_system.process_frame(frame)
        alerts.append(alert)
        probabilities.append(prob)
        
        #add visual alert to frame
        if alert:
            #red border for alert
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
            cv2.putText(frame, f"BUMP AHEAD! ({prob:.0%})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Clear ({prob:.0%})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if out:
            out.write(frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"processed {frame_idx} frames...")
    
    cap.release()
    if out:
        out.release()
    
    return alerts, probabilities

#run demo
if video_files:
    print("running alert system demo...")
    alerts, probs = demo_alert_system(
        video_files[0], 
        OUTPUT_DIR / 'bump_predictor.joblib',
        OUTPUT_DIR / 'bump_alert_demo.mp4'
    )
    print(f"\ndemo complete! output saved to {OUTPUT_DIR / 'bump_alert_demo.mp4'}")
    print(f"total alerts triggered: {sum(alerts)} frames")


#visualize alert predictions over time
if 'probs' in dir():
    time_probs = np.arange(len(probs)) / FPS
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    #probability over time
    axes[0].plot(time_probs, probs, 'b-', alpha=0.7)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='alert threshold')
    axes[0].fill_between(time_probs, 0, probs, where=np.array(probs) > 0.5, 
                         color='red', alpha=0.3, label='alert active')
    axes[0].set_ylabel('bump probability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #compare with actual bumps
    axes[1].plot(time_signal, bump_signal, 'g-', alpha=0.7, label='actual bump signal')
    axes[1].axhline(y=bump_threshold, color='orange', linestyle='--', label='detection threshold')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('bump intensity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('bump prediction vs actual bumps')
    plt.tight_layout()
    plt.show()


#visualize frames before and during bumps
def visualize_bump_sequence(frames, flows, bump_frame, pre_frames=10, post_frames=5):
    """show frames leading up to and including a bump"""
    start = max(0, bump_frame - pre_frames)
    end = min(len(frames), bump_frame + post_frames + 1)
    
    sequence = frames[start:end]
    n_frames = len(sequence)
    
    fig, axes = plt.subplots(2, n_frames, figsize=(2*n_frames, 5))
    
    for i, frame in enumerate(sequence):
        frame_idx = start + i
        
        #original frame
        axes[0, i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        title = f"t={frame_idx}"
        if frame_idx == bump_frame:
            title += " (BUMP)"
            axes[0, i].set_title(title, color='red', fontweight='bold')
        else:
            axes[0, i].set_title(title)
        axes[0, i].axis('off')
        
        #edge detection (shows cracks)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        axes[1, i].imshow(edges, cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('original')
    axes[1, 0].set_ylabel('edges')
    
    plt.suptitle(f'bump at frame {bump_frame}')
    plt.tight_layout()
    return fig

#visualize first few bumps
n_bumps_to_show = min(3, len(bump_frames))
for i, bf in enumerate(bump_frames[:n_bumps_to_show]):
    print(f"\nbump {i+1} at frame {bf} (time: {bf/FPS:.2f}s)")
    visualize_bump_sequence(frames, flows, bf)
    plt.show()


#analyze texture features before vs during/after bumps
def compare_bump_features(frames, bump_frames, pre_window=10):
    """compare texture features before bumps vs random segments"""
    bump_features = []
    normal_features = []
    
    #features before bumps
    for bf in bump_frames:
        start = max(0, bf - pre_window)
        for frame in frames[start:bf]:
            bump_features.append(extract_texture_features(frame))
    
    #random non-bump segments
    bump_set = set()
    for bf in bump_frames:
        bump_set.update(range(max(0, bf-pre_window), min(len(frames), bf+5)))
    
    normal_indices = [i for i in range(len(frames)) if i not in bump_set]
    np.random.seed(42)
    sample_indices = np.random.choice(normal_indices, size=min(len(normal_indices), len(bump_features)), replace=False)
    
    for idx in sample_indices:
        normal_features.append(extract_texture_features(frames[idx]))
    
    return bump_features, normal_features

if len(bump_frames) > 0:
    print("analyzing texture differences...")
    bump_feats, normal_feats = compare_bump_features(frames, bump_frames)
    
    #convert to arrays for comparison
    feature_keys = list(bump_feats[0].keys())
    bump_arr = np.array([[f[k] for k in feature_keys] for f in bump_feats])
    normal_arr = np.array([[f[k] for k in feature_keys] for f in normal_feats])
    
    #compare means
    print("\nfeature comparison (before bump vs normal):")
    print(f"{'feature':<20} {'bump mean':>12} {'normal mean':>12} {'diff':>12}")
    print("-" * 60)
    
    for i, key in enumerate(feature_keys):
        bump_mean = bump_arr[:, i].mean()
        normal_mean = normal_arr[:, i].mean()
        diff = bump_mean - normal_mean
        diff_pct = 100 * diff / (normal_mean + 1e-6)
        print(f"{key:<20} {bump_mean:>12.4f} {normal_mean:>12.4f} {diff_pct:>+10.1f}%")


#summary
print("="*60)
print("BUMP DETECTION & PREDICTION SUMMARY")
print("="*60)
print(f"\nvideo analyzed: {video_path.name if video_files else 'N/A'}")
print(f"total frames: {len(frames)}")
print(f"video duration: {len(frames)/FPS:.1f} seconds")
print(f"\nbumps detected: {len(bump_frames)}")
print(f"bump detection threshold: {bump_threshold:.3f}")
print(f"\nmodel trained on: {len(X)} segments")
print(f"segment length: {SEGMENT_LENGTH} frames")
print(f"lookahead window: {LOOKAHEAD_FRAMES} frames")
print(f"\nmodel saved to: {OUTPUT_DIR / 'bump_predictor.joblib'}")
print(f"\nto use the alert system:")
print("  alert_system = BumpAlertSystem('output/bump_predictor.joblib')")
print("  alert, prob = alert_system.process_frame(frame)")
