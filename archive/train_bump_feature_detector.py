#train a model to detect visual features that CAUSE bumps
#only highlights features that are predictive of upcoming bumps
import cv2
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_DIR = Path("data-scaled")
OUTPUT_DIR = Path("output")
FPS = 20
GRID_SIZE = 8  #divide frame into 8x8 grid for region-based detection

def load_data():
    """load video and bump detection results"""
    video_files = list(DATA_DIR.glob("*.mp4"))
    video_path = video_files[0]
    
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    with open(OUTPUT_DIR / 'bump_detection_robust.pkl', 'rb') as f:
        detection = pickle.load(f)
    
    return np.array(frames), detection['bump_frames'], detection['bump_intensities']

def extract_region_features(gray, x, y, w, h):
    """extract features for a single region"""
    region = gray[y:y+h, x:x+w]
    if region.size == 0:
        return None
    
    features = {}
    
    #intensity features
    features['intensity_mean'] = region.mean()
    features['intensity_std'] = region.std()
    
    #edge features
    edges = cv2.Canny(region, 50, 150)
    features['edge_density'] = edges.mean() / 255
    
    #gradient features
    sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    features['grad_x'] = np.abs(sobel_x).mean()
    features['grad_y'] = np.abs(sobel_y).mean()
    features['grad_ratio'] = features['grad_y'] / (features['grad_x'] + 1e-6)
    
    #texture (laplacian)
    laplacian = cv2.Laplacian(region, cv2.CV_64F)
    features['texture'] = laplacian.var()
    
    #line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=w//3, maxLineGap=5)
    features['n_lines'] = len(lines) if lines is not None else 0
    
    #horizontal line score
    h_line_score = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 20 or angle > 160:  #horizontal
                h_line_score += 1
    features['h_lines'] = h_line_score
    
    #contrast
    features['contrast'] = region.max() - region.min()
    
    return features

def extract_grid_features(frame, grid_size=GRID_SIZE):
    """extract features for each grid cell"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    #focus on bottom 2/3 (road ahead)
    roi_start = h // 3
    roi_h = h - roi_start
    
    cell_h = roi_h // grid_size
    cell_w = w // grid_size
    
    grid_features = []
    grid_positions = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_w
            y = roi_start + row * cell_h
            
            feat = extract_region_features(gray, x, y, cell_w, cell_h)
            if feat:
                #add position features
                feat['row'] = row / grid_size
                feat['col'] = col / grid_size
                feat['dist_from_center'] = np.sqrt((row/grid_size - 0.5)**2 + (col/grid_size - 0.5)**2)
                
                grid_features.append(feat)
                grid_positions.append((x, y, cell_w, cell_h))
    
    return grid_features, grid_positions

def create_training_data(frames, bump_frames, lookahead_start=5, lookahead_end=15):
    """create training data: regions before bumps vs random regions"""
    print("creating training data...")
    
    X = []
    y = []
    
    bump_set = set(bump_frames)
    feature_names = None
    
    #collect positive samples: regions in frames before bumps
    for bump_frame in bump_frames:
        #frames leading up to bump
        for offset in range(lookahead_start, lookahead_end):
            frame_idx = bump_frame - offset
            if frame_idx < 0 or frame_idx >= len(frames):
                continue
            
            frame = frames[frame_idx]
            grid_features, grid_positions = extract_grid_features(frame)
            
            #the features in the BOTTOM rows are more likely bump-causing
            #weight by row position (bottom = more important)
            for i, feat in enumerate(grid_features):
                row = int(feat['row'] * GRID_SIZE)
                
                #bottom 2 rows are most relevant for bump causation
                if row >= GRID_SIZE - 2:
                    if feature_names is None:
                        feature_names = list(feat.keys())
                    X.append([feat[k] for k in feature_names])
                    y.append(1)  #bump-causing region
    
    print(f"  positive samples: {sum(y)}")
    
    #collect negative samples: regions from frames NOT near bumps
    negative_frames = []
    for i in range(len(frames)):
        #check if far from any bump
        is_near_bump = False
        for bf in bump_frames:
            if abs(i - bf) < 30:  #within 30 frames of bump
                is_near_bump = True
                break
        if not is_near_bump:
            negative_frames.append(i)
    
    #sample negative frames
    np.random.seed(42)
    n_negative_needed = sum(y) * 2  #2x negative samples
    sampled_negatives = np.random.choice(negative_frames, size=min(n_negative_needed // (GRID_SIZE * 2), len(negative_frames)), replace=False)
    
    for frame_idx in sampled_negatives:
        frame = frames[frame_idx]
        grid_features, _ = extract_grid_features(frame)
        
        for feat in grid_features:
            X.append([feat[k] for k in feature_names])
            y.append(0)  #not bump-causing
    
    print(f"  negative samples: {len(y) - sum(y)}")
    print(f"  total: {len(y)}")
    
    return np.array(X), np.array(y), feature_names

def train_model(X, y):
    """train bump-causing feature classifier"""
    print("\ntraining model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #train gradient boosting (good for this task)
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    #evaluate
    y_pred = model.predict(X_test_scaled)
    print("\nmodel evaluation:")
    print(classification_report(y_test, y_pred, target_names=['normal', 'bump-causing']))
    
    return model, scaler

def main():
    print("loading data...")
    frames, bump_frames, bump_intensities = load_data()
    print(f"  {len(frames)} frames, {len(bump_frames)} bumps")
    
    #create training data
    X, y, feature_names = create_training_data(frames, bump_frames)
    
    #train model
    model, scaler = train_model(X, y)
    
    #save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'grid_size': GRID_SIZE
    }
    joblib.dump(model_data, OUTPUT_DIR / 'bump_feature_detector.joblib')
    print(f"\nmodel saved to {OUTPUT_DIR / 'bump_feature_detector.joblib'}")
    
    #print feature importance
    print("\nfeature importance:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(len(feature_names)):
        print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

if __name__ == "__main__":
    main()




