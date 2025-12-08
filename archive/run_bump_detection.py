#bump detection and prediction pipeline
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

#config
DATA_DIR = Path("data-scaled")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS = 20
SEGMENT_LENGTH = 10
BUMP_THRESHOLD_PERCENTILE = 95
LOOKAHEAD_FRAMES = 10

def load_video(video_path):
    """load video frames"""
    print(f"loading {video_path.name}...")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) % 1000 == 0:
            print(f"  loaded {len(frames)}/{total} frames")
    
    cap.release()
    print(f"  loaded {len(frames)} frames, shape: {frames[0].shape}")
    return np.array(frames)

def compute_optical_flow(frames):
    """compute dense optical flow"""
    print("computing optical flow...")
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    flows = []
    
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(flow)
        
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(grays) - 1} frames")
    
    print(f"  computed {len(flows)} flow fields")
    return flows

def extract_flow_features(flows):
    """extract vertical flow statistics"""
    features = {
        'avg_vy': [], 'std_vy': [], 'max_vy': [], 'min_vy': [],
        'avg_vx': [], 'magnitude': [], 'bottom_vy': [],
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
        features['bottom_vy'].append(vy[h//2:, :].mean())
    
    return {k: np.array(v) for k, v in features.items()}

def detect_bumps(flow_features, threshold_percentile=95, min_distance=5):
    """detect bumps from vertical flow changes"""
    avg_vy = flow_features['avg_vy']
    vy_diff = np.abs(np.diff(avg_vy))
    vy_abs = np.abs(avg_vy[1:])
    bump_signal = vy_diff + 0.5 * vy_abs
    bump_signal_smooth = gaussian_filter1d(bump_signal, sigma=1)
    threshold = np.percentile(bump_signal_smooth, threshold_percentile)
    
    peaks, properties = signal.find_peaks(
        bump_signal_smooth, height=threshold, distance=min_distance
    )
    
    bump_frames = peaks + 1
    bump_intensities = properties['peak_heights']
    return bump_frames, bump_intensities, bump_signal_smooth, threshold

def extract_texture_features(frame):
    """extract texture features for bump prediction"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    roi = gray[h//2:, :]
    
    features = {}
    
    edges = cv2.Canny(roi, 50, 150)
    features['edge_density'] = edges.mean() / 255
    features['edge_std'] = edges.std() / 255
    
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    features['sobel_x_mean'] = np.abs(sobel_x).mean()
    features['sobel_y_mean'] = np.abs(sobel_y).mean()
    features['sobel_ratio'] = features['sobel_y_mean'] / (features['sobel_x_mean'] + 1e-6)
    
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    features['laplacian_var'] = laplacian.var()
    features['laplacian_mean'] = np.abs(laplacian).mean()
    
    features['intensity_mean'] = roi.mean()
    features['intensity_std'] = roi.std()
    
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

def extract_segment_features(frames, flows, start_idx, segment_length=10):
    """extract features from a segment"""
    end_idx = min(start_idx + segment_length, len(frames))
    segment_frames = frames[start_idx:end_idx]
    segment_flows = flows[start_idx:min(end_idx, len(flows))]
    
    if len(segment_frames) < segment_length // 2:
        return None
    
    features = {}
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

def create_training_data(frames, flows, bump_frames, segment_length=10, lookahead=10):
    """create training dataset"""
    print("creating training data...")
    X, y, segment_indices = [], [], []
    bump_set = set(bump_frames)
    step = segment_length // 2
    
    for start_idx in range(0, len(frames) - segment_length - lookahead, step):
        features = extract_segment_features(frames, flows, start_idx, segment_length)
        if features is None:
            continue
        
        lookahead_start = start_idx + segment_length
        lookahead_end = lookahead_start + lookahead
        has_bump = any(bf in range(lookahead_start, lookahead_end) for bf in bump_set)
        
        X.append(list(features.values()))
        y.append(1 if has_bump else 0)
        segment_indices.append(start_idx)
        
        if len(X) % 200 == 0:
            print(f"  processed {len(X)} segments...")
    
    feature_names = list(features.keys()) if features else []
    return np.array(X), np.array(y), segment_indices, feature_names

def main():
    #find video
    video_files = list(DATA_DIR.glob("*.mp4")) + list(DATA_DIR.glob("*.mov"))
    if not video_files:
        print("no videos found in data-scaled/")
        return
    
    video_path = video_files[0]
    print(f"processing: {video_path.name}")
    
    #load and process
    frames = load_video(video_path)
    flows = compute_optical_flow(frames)
    
    #extract flow features and detect bumps
    print("\nextracting flow features...")
    flow_features = extract_flow_features(flows)
    
    print("detecting bumps...")
    bump_frames, bump_intensities, bump_signal, bump_threshold = detect_bumps(
        flow_features, threshold_percentile=BUMP_THRESHOLD_PERCENTILE
    )
    print(f"  detected {len(bump_frames)} bumps")
    print(f"  threshold: {bump_threshold:.3f}")
    
    #create labels
    num_frames = len(frames)
    bump_labels = np.zeros(num_frames, dtype=int)
    for bf in bump_frames:
        start = max(0, bf - 2)
        end = min(num_frames, bf + 3)
        bump_labels[start:end] = 1
    
    print(f"\ntotal frames: {num_frames}")
    print(f"bump frames: {bump_labels.sum()} ({100*bump_labels.mean():.1f}%)")
    
    #create training data
    X, y, segment_indices, feature_names = create_training_data(
        frames, flows, bump_frames, 
        segment_length=SEGMENT_LENGTH, 
        lookahead=LOOKAHEAD_FRAMES
    )
    
    print(f"\ntraining data: X={X.shape}, y={y.shape}")
    print(f"positive samples: {y.sum()} ({100*y.mean():.1f}%)")
    
    #train model
    print("\ntraining model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        class_weight=class_weight_dict, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    #evaluate
    y_pred = rf_model.predict(X_test_scaled)
    print("\nmodel evaluation:")
    print(classification_report(y_test, y_pred, target_names=['no bump', 'bump']))
    
    #feature importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\ntop 10 features:")
    for i in range(min(10, len(feature_names))):
        print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    #save model
    model_data = {
        'model': rf_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'segment_length': SEGMENT_LENGTH,
        'lookahead': LOOKAHEAD_FRAMES
    }
    joblib.dump(model_data, OUTPUT_DIR / 'bump_predictor.joblib')
    print(f"\nmodel saved to {OUTPUT_DIR / 'bump_predictor.joblib'}")
    
    #save detection results
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
    
    #plot bump detection
    print("\ngenerating plots...")
    time_signal = np.arange(len(bump_signal)) / FPS
    
    plt.figure(figsize=(14, 5))
    plt.plot(time_signal, bump_signal, 'b-', alpha=0.7, label='bump signal')
    plt.axhline(y=bump_threshold, color='r', linestyle='--', label=f'threshold ({BUMP_THRESHOLD_PERCENTILE}th %ile)')
    bump_times = (bump_frames - 1) / FPS
    plt.scatter(bump_times, bump_intensities, c='red', s=100, marker='v', label='detected bumps', zorder=5)
    plt.xlabel('time (s)')
    plt.ylabel('bump signal intensity')
    plt.title('bump detection from vertical optical flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bump_detection_plot.png', dpi=150)
    print(f"  saved {OUTPUT_DIR / 'bump_detection_plot.png'}")
    
    #plot flow features
    time = np.arange(len(flow_features['avg_vy'])) / FPS
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time, flow_features['avg_vy'], 'b-', alpha=0.7)
    axes[0].set_ylabel('avg vy')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time, flow_features['std_vy'], 'r-', alpha=0.7)
    axes[1].set_ylabel('std vy')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(time, flow_features['magnitude'], 'g-', alpha=0.7)
    axes[2].set_ylabel('magnitude')
    axes[2].set_xlabel('time (s)')
    axes[2].grid(True, alpha=0.3)
    plt.suptitle('optical flow over time')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optical_flow_plot.png', dpi=150)
    print(f"  saved {OUTPUT_DIR / 'optical_flow_plot.png'}")
    
    #feature importance plot
    plt.figure(figsize=(10, 8))
    top_n = min(20, len(feature_names))
    plt.barh(range(top_n), importances[indices[:top_n]][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('importance')
    plt.title('top features for bump prediction')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150)
    print(f"  saved {OUTPUT_DIR / 'feature_importance.png'}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"video: {video_path.name}")
    print(f"duration: {len(frames)/FPS:.1f} seconds")
    print(f"bumps detected: {len(bump_frames)}")
    print(f"model accuracy: see classification report above")
    print(f"\noutput files in {OUTPUT_DIR}/:")
    print("  - bump_predictor.joblib (trained model)")
    print("  - bump_detection.pkl (detection results)")
    print("  - bump_detection_plot.png")
    print("  - optical_flow_plot.png")
    print("  - feature_importance.png")

if __name__ == "__main__":
    main()




