#advanced bump detection with multiple ML models and enhanced features
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, make_scorer
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE
import joblib

#try importing optional libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed, skipping XGBoost model")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("lightgbm not installed, skipping LightGBM model")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imbalanced-learn not installed, using class weights instead")

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
        if len(frames) % 2000 == 0:
            print(f"  loaded {len(frames)}/{total} frames")
    
    cap.release()
    print(f"  loaded {len(frames)} frames")
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
        if (i + 1) % 2000 == 0:
            print(f"  processed {i + 1}/{len(grays) - 1} frames")
    
    return flows

def extract_flow_features(flows):
    """extract flow statistics"""
    features = {
        'avg_vy': [], 'std_vy': [], 'max_vy': [], 'min_vy': [],
        'avg_vx': [], 'std_vx': [], 'magnitude': [], 
        'bottom_vy': [], 'top_vy': [], 'center_vy': [],
    }
    
    for flow in flows:
        vx, vy = flow[..., 0], flow[..., 1]
        h = flow.shape[0]
        
        features['avg_vy'].append(vy.mean())
        features['std_vy'].append(vy.std())
        features['max_vy'].append(vy.max())
        features['min_vy'].append(vy.min())
        features['avg_vx'].append(vx.mean())
        features['std_vx'].append(vx.std())
        features['magnitude'].append(np.sqrt(vx**2 + vy**2).mean())
        features['bottom_vy'].append(vy[2*h//3:, :].mean())
        features['top_vy'].append(vy[:h//3, :].mean())
        features['center_vy'].append(vy[h//3:2*h//3, :].mean())
    
    return {k: np.array(v) for k, v in features.items()}

def detect_bumps(flow_features, threshold_percentile=95, min_distance=5):
    """detect bumps from vertical flow"""
    avg_vy = flow_features['avg_vy']
    vy_diff = np.abs(np.diff(avg_vy))
    vy_abs = np.abs(avg_vy[1:])
    bump_signal = vy_diff + 0.5 * vy_abs
    bump_signal_smooth = gaussian_filter1d(bump_signal, sigma=1)
    threshold = np.percentile(bump_signal_smooth, threshold_percentile)
    
    peaks, properties = signal.find_peaks(
        bump_signal_smooth, height=threshold, distance=min_distance
    )
    
    return peaks + 1, properties['peak_heights'], bump_signal_smooth, threshold

#=== ENHANCED FEATURE EXTRACTION ===

def extract_texture_features(frame):
    """basic texture features"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    roi = gray[h//2:, :]
    
    features = {}
    
    #edge features
    edges = cv2.Canny(roi, 50, 150)
    features['edge_density'] = edges.mean() / 255
    features['edge_std'] = edges.std() / 255
    
    #sobel gradients
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    features['sobel_x_mean'] = np.abs(sobel_x).mean()
    features['sobel_y_mean'] = np.abs(sobel_y).mean()
    features['sobel_ratio'] = features['sobel_y_mean'] / (features['sobel_x_mean'] + 1e-6)
    
    #laplacian
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    features['laplacian_var'] = laplacian.var()
    features['laplacian_mean'] = np.abs(laplacian).mean()
    
    #intensity
    features['intensity_mean'] = roi.mean()
    features['intensity_std'] = roi.std()
    features['intensity_skew'] = ((roi - roi.mean()) ** 3).mean() / (roi.std() ** 3 + 1e-6)
    
    #gabor filters
    for i, theta in enumerate([0, np.pi/4, np.pi/2, 3*np.pi/4]):
        kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0)
        filtered = cv2.filter2D(roi, cv2.CV_64F, kernel)
        features[f'gabor_{i}'] = np.abs(filtered).mean()
    
    return features

def extract_hog_features(frame, cell_size=8):
    """histogram of oriented gradients features"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    roi = gray[h//2:, :]
    
    #compute gradients
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi % 180
    
    #simple hog: histogram of gradient orientations
    n_bins = 9
    hist, _ = np.histogram(angle, bins=n_bins, range=(0, 180), weights=magnitude)
    hist = hist / (hist.sum() + 1e-6)
    
    features = {f'hog_{i}': hist[i] for i in range(n_bins)}
    features['hog_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
    
    return features

def extract_lbp_features(frame):
    """local binary pattern features (simplified)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    roi = gray[h//2:, :]
    
    #simple lbp: compare center pixel with neighbors
    lbp = np.zeros_like(roi, dtype=np.uint8)
    for i in range(1, roi.shape[0]-1):
        for j in range(1, roi.shape[1]-1):
            center = roi[i, j]
            code = 0
            code |= (roi[i-1, j-1] > center) << 7
            code |= (roi[i-1, j] > center) << 6
            code |= (roi[i-1, j+1] > center) << 5
            code |= (roi[i, j+1] > center) << 4
            code |= (roi[i+1, j+1] > center) << 3
            code |= (roi[i+1, j] > center) << 2
            code |= (roi[i+1, j-1] > center) << 1
            code |= (roi[i, j-1] > center) << 0
            lbp[i, j] = code
    
    #histogram of lbp
    hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
    hist = hist / (hist.sum() + 1e-6)
    
    features = {f'lbp_{i}': hist[i] for i in range(len(hist))}
    features['lbp_uniformity'] = np.sum(hist ** 2)
    
    return features

def extract_frequency_features(segment_flows):
    """frequency domain features from optical flow"""
    if len(segment_flows) < 4:
        return {}
    
    vy_series = np.array([f[..., 1].mean() for f in segment_flows])
    
    #fft
    fft_vals = np.abs(fft(vy_series))
    n = len(fft_vals)
    
    features = {}
    features['fft_dc'] = fft_vals[0]
    features['fft_low'] = fft_vals[1:n//4].mean() if n > 4 else 0
    features['fft_mid'] = fft_vals[n//4:n//2].mean() if n > 4 else 0
    features['fft_high'] = fft_vals[n//2:].mean() if n > 2 else 0
    features['fft_peak_freq'] = np.argmax(fft_vals[1:n//2]) + 1 if n > 2 else 0
    features['fft_energy'] = np.sum(fft_vals ** 2)
    
    return features

def extract_temporal_features(segment_flows):
    """temporal dynamics features"""
    if len(segment_flows) < 3:
        return {}
    
    vy = np.array([f[..., 1].mean() for f in segment_flows])
    vx = np.array([f[..., 0].mean() for f in segment_flows])
    mag = np.array([np.sqrt(f[..., 0]**2 + f[..., 1]**2).mean() for f in segment_flows])
    
    features = {}
    
    #derivatives
    vy_diff = np.diff(vy)
    vy_diff2 = np.diff(vy_diff) if len(vy_diff) > 1 else np.array([0])
    
    features['vy_velocity'] = vy_diff.mean()
    features['vy_acceleration'] = vy_diff2.mean()
    features['vy_jerk'] = np.diff(vy_diff2).mean() if len(vy_diff2) > 1 else 0
    
    #autocorrelation
    if len(vy) > 2:
        autocorr = np.correlate(vy - vy.mean(), vy - vy.mean(), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        features['autocorr_1'] = autocorr[1] if len(autocorr) > 1 else 0
        features['autocorr_2'] = autocorr[2] if len(autocorr) > 2 else 0
    
    #zero crossings
    features['zero_crossings'] = np.sum(np.diff(np.sign(vy - vy.mean())) != 0)
    
    #peaks
    peaks, _ = signal.find_peaks(np.abs(vy))
    features['n_peaks'] = len(peaks)
    
    #range and variability
    features['vy_range'] = vy.max() - vy.min()
    features['mag_range'] = mag.max() - mag.min()
    features['vy_iqr'] = np.percentile(vy, 75) - np.percentile(vy, 25)
    
    return features

def extract_segment_features_advanced(frames, flows, start_idx, segment_length=10):
    """extract comprehensive features from segment"""
    end_idx = min(start_idx + segment_length, len(frames))
    segment_frames = frames[start_idx:end_idx]
    segment_flows = flows[start_idx:min(end_idx, len(flows))]
    
    if len(segment_frames) < segment_length // 2:
        return None
    
    features = {}
    
    #aggregate texture features
    texture_feats = defaultdict(list)
    hog_feats = defaultdict(list)
    
    for frame in segment_frames:
        tf = extract_texture_features(frame)
        for k, v in tf.items():
            texture_feats[k].append(v)
        
        hf = extract_hog_features(frame)
        for k, v in hf.items():
            hog_feats[k].append(v)
    
    #texture aggregation
    for k, v in texture_feats.items():
        features[f'tex_{k}_mean'] = np.mean(v)
        features[f'tex_{k}_std'] = np.std(v)
        features[f'tex_{k}_max'] = np.max(v)
        features[f'tex_{k}_min'] = np.min(v)
        features[f'tex_{k}_trend'] = v[-1] - v[0] if len(v) > 1 else 0
        features[f'tex_{k}_range'] = np.max(v) - np.min(v)
    
    #hog aggregation
    for k, v in hog_feats.items():
        features[f'{k}_mean'] = np.mean(v)
        features[f'{k}_std'] = np.std(v)
    
    #lbp from middle frame
    mid_idx = len(segment_frames) // 2
    lbp_feats = extract_lbp_features(segment_frames[mid_idx])
    features.update(lbp_feats)
    
    #optical flow features
    if len(segment_flows) > 0:
        vy_vals = [flow[..., 1].mean() for flow in segment_flows]
        vx_vals = [flow[..., 0].mean() for flow in segment_flows]
        mag_vals = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean() for flow in segment_flows]
        
        features['flow_vy_mean'] = np.mean(vy_vals)
        features['flow_vy_std'] = np.std(vy_vals)
        features['flow_vy_max'] = np.max(vy_vals)
        features['flow_vy_min'] = np.min(vy_vals)
        features['flow_vy_trend'] = vy_vals[-1] - vy_vals[0] if len(vy_vals) > 1 else 0
        features['flow_vx_mean'] = np.mean(vx_vals)
        features['flow_vx_std'] = np.std(vx_vals)
        features['flow_mag_mean'] = np.mean(mag_vals)
        features['flow_mag_max'] = np.max(mag_vals)
        features['flow_mag_std'] = np.std(mag_vals)
        
        #frequency features
        freq_feats = extract_frequency_features(segment_flows)
        features.update(freq_feats)
        
        #temporal features
        temp_feats = extract_temporal_features(segment_flows)
        features.update(temp_feats)
    
    return features

def create_training_data(frames, flows, bump_frames, segment_length=10, lookahead=10):
    """create training dataset with advanced features"""
    print("creating training data with advanced features...")
    X, y, segment_indices = [], [], []
    bump_set = set(bump_frames)
    step = segment_length // 2
    
    for start_idx in range(0, len(frames) - segment_length - lookahead, step):
        features = extract_segment_features_advanced(frames, flows, start_idx, segment_length)
        if features is None:
            continue
        
        lookahead_start = start_idx + segment_length
        lookahead_end = lookahead_start + lookahead
        has_bump = any(bf in range(lookahead_start, lookahead_end) for bf in bump_set)
        
        X.append(list(features.values()))
        y.append(1 if has_bump else 0)
        segment_indices.append(start_idx)
        
        if len(X) % 500 == 0:
            print(f"  processed {len(X)} segments...")
    
    feature_names = list(features.keys()) if features else []
    return np.array(X), np.array(y), segment_indices, feature_names

#=== MODEL TRAINING ===

def train_models(X_train, y_train, X_test, y_test, class_weight_dict):
    """train multiple models and return results"""
    results = {}
    
    #base models
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            class_weight=class_weight_dict, random_state=42, n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=200, max_depth=15,
            class_weight=class_weight_dict, random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'adaboost': AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=42
        ),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, random_state=42
        ),
        'svm': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight=class_weight_dict, probability=True, random_state=42
        ),
    }
    
    if HAS_XGB:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        models['xgboost'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'
        )
    
    if HAS_LGBM:
        models['lightgbm'] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            class_weight=class_weight_dict,
            random_state=42, n_jobs=-1, verbose=-1
        )
    
    #train and evaluate each model
    for name, model in models.items():
        print(f"\ntraining {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
            
            results[name] = {
                'model': model,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            print(f"  F1: {f1:.4f}, AUC: {auc:.4f}")
        except Exception as e:
            print(f"  error: {e}")
    
    return results

def create_ensemble(results, X_train, y_train, X_test, y_test):
    """create ensemble from best models"""
    print("\ncreating ensemble models...")
    
    #select top models by f1
    sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    top_models = [(name, res['model']) for name, res in sorted_models[:5]]
    
    print(f"  top models: {[m[0] for m in top_models]}")
    
    #voting ensemble (soft voting)
    voting = VotingClassifier(
        estimators=top_models,
        voting='soft'
    )
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)
    y_proba = voting.predict_proba(X_test)[:, 1]
    
    results['voting_ensemble'] = {
        'model': voting,
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'predictions': y_pred,
        'probabilities': y_proba
    }
    print(f"  voting ensemble F1: {results['voting_ensemble']['f1']:.4f}")
    
    #stacking ensemble
    stacking = StackingClassifier(
        estimators=top_models[:4],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)[:, 1]
    
    results['stacking_ensemble'] = {
        'model': stacking,
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'predictions': y_pred,
        'probabilities': y_proba
    }
    print(f"  stacking ensemble F1: {results['stacking_ensemble']['f1']:.4f}")
    
    return results

def optimize_threshold(y_test, y_proba):
    """find optimal prediction threshold"""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold, f1_scores[best_idx]

def main():
    #find video
    video_files = list(DATA_DIR.glob("*.mp4")) + list(DATA_DIR.glob("*.mov"))
    if not video_files:
        print("no videos found!")
        return
    
    video_path = video_files[0]
    print(f"processing: {video_path.name}\n")
    
    #load data
    frames = load_video(video_path)
    flows = compute_optical_flow(frames)
    
    #detect bumps
    print("\nextracting flow features...")
    flow_features = extract_flow_features(flows)
    
    print("detecting bumps...")
    bump_frames, bump_intensities, bump_signal, bump_threshold = detect_bumps(
        flow_features, threshold_percentile=BUMP_THRESHOLD_PERCENTILE
    )
    print(f"  detected {len(bump_frames)} bumps")
    
    #create training data
    X, y, segment_indices, feature_names = create_training_data(
        frames, flows, bump_frames, 
        segment_length=SEGMENT_LENGTH, 
        lookahead=LOOKAHEAD_FRAMES
    )
    
    print(f"\ntraining data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"positive samples: {y.sum()} ({100*y.mean():.1f}%)")
    
    #split data (time-aware - no shuffling for temporal data)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    
    #scale features
    scaler = RobustScaler()  #more robust to outliers
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #handle class imbalance
    if HAS_IMBLEARN:
        print("\napplying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"  balanced: {len(X_train_balanced)} samples")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    #compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    #train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    results = train_models(X_train_balanced, y_train_balanced, X_test_scaled, y_test, class_weight_dict)
    results = create_ensemble(results, X_train_balanced, y_train_balanced, X_test_scaled, y_test)
    
    #find best model
    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_result = results[best_name]
    print(f"\nbest model: {best_name} (F1: {best_result['f1']:.4f})")
    
    #optimize threshold
    opt_threshold, opt_f1 = optimize_threshold(y_test, best_result['probabilities'])
    print(f"optimal threshold: {opt_threshold:.3f} (F1: {opt_f1:.4f})")
    
    #final evaluation
    print("\n" + "="*60)
    print(f"FINAL EVALUATION ({best_name})")
    print("="*60)
    
    y_pred_opt = (best_result['probabilities'] >= opt_threshold).astype(int)
    print(classification_report(y_test, y_pred_opt, target_names=['no bump', 'bump']))
    
    #comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'F1':>8} {'AUC':>8}")
    print("-" * 45)
    for name, res in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:<25} {res['f1']:>8.4f} {res['auc']:>8.4f}")
    
    #save best model
    model_data = {
        'model': best_result['model'],
        'scaler': scaler,
        'feature_names': feature_names,
        'segment_length': SEGMENT_LENGTH,
        'lookahead': LOOKAHEAD_FRAMES,
        'optimal_threshold': opt_threshold,
        'model_name': best_name
    }
    joblib.dump(model_data, OUTPUT_DIR / 'bump_predictor_advanced.joblib')
    print(f"\nmodel saved to {OUTPUT_DIR / 'bump_predictor_advanced.joblib'}")
    
    #save all results
    all_results = {name: {'f1': r['f1'], 'auc': r['auc']} for name, r in results.items()}
    with open(OUTPUT_DIR / 'model_comparison.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    #plot comparison
    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    f1s = [results[n]['f1'] for n in names]
    aucs = [results[n]['auc'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, f1s, width, label='F1 Score', color='steelblue')
    plt.bar(x + width/2, aucs, width, label='AUC', color='coral')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150)
    print(f"saved {OUTPUT_DIR / 'model_comparison.png'}")
    
    #feature importance from best tree-based model
    for name in ['xgboost', 'lightgbm', 'random_forest', 'extra_trees']:
        if name in results and hasattr(results[name]['model'], 'feature_importances_'):
            importances = results[name]['model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 10))
            top_n = min(30, len(feature_names))
            plt.barh(range(top_n), importances[indices[:top_n]][::-1])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1], fontsize=8)
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Features ({name})')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'feature_importance_advanced.png', dpi=150)
            print(f"saved {OUTPUT_DIR / 'feature_importance_advanced.png'}")
            
            print(f"\ntop 15 features ({name}):")
            for i in range(min(15, len(feature_names))):
                print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            break
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()




