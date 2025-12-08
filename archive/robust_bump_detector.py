#robust bump detector - combines visual defect detection with motion analysis
#designed to work with OIS-stabilized phone video
import cv2
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.fft import fft, fftfreq
import pickle
import matplotlib.pyplot as plt
from collections import deque

DATA_DIR = Path("data-scaled")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
FPS = 20

#=== VISUAL DEFECT DETECTION ===

def detect_horizontal_lines(gray, roi_start):
    """detect horizontal lines/cracks that indicate bumps or joints"""
    roi = gray[roi_start:, :]
    h, w = roi.shape
    
    #sobel for horizontal edges
    sobel_h = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    sobel_h_abs = np.abs(sobel_h)
    
    #hough line detection
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=w//4, maxLineGap=10)
    
    horizontal_score = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            #horizontal lines have angle near 0 or 180
            if angle < 15 or angle > 165:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                horizontal_score += length / w
    
    #also measure overall horizontal edge strength
    horizontal_strength = np.mean(sobel_h_abs[h//2:, :])  #bottom half stronger weight
    
    return horizontal_score, horizontal_strength

def detect_texture_roughness(gray, roi_start):
    """detect rough texture patterns indicating bumpy surface"""
    roi = gray[roi_start:, :]
    h, w = roi.shape
    
    #laplacian variance (higher = more texture)
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    texture_var = laplacian.var()
    
    #local binary pattern-like measure
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    high_pass = cv2.filter2D(roi, cv2.CV_64F, kernel)
    roughness = np.abs(high_pass).mean()
    
    #bottom region (closer to scooter)
    bottom_roughness = np.abs(high_pass[2*h//3:, :]).mean()
    
    return texture_var, roughness, bottom_roughness

def detect_shadow_lines(gray, roi_start):
    """detect shadow lines that indicate height changes/bumps"""
    roi = gray[roi_start:, :]
    h, w = roi.shape
    
    #gradient in vertical direction
    grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=5)
    
    #look for strong negative-to-positive transitions (shadow edge)
    shadow_score = 0
    for row in range(h//2, h):  #bottom half
        row_grad = grad_y[row, :]
        #find transitions
        transitions = np.abs(np.diff(np.sign(row_grad))).sum()
        shadow_score += transitions
    
    shadow_score /= (h//2 * w)  #normalize
    return shadow_score

def detect_dark_regions(gray, roi_start):
    """detect unusually dark regions (potholes, cracks)"""
    roi = gray[roi_start:, :]
    
    #adaptive threshold to find dark regions
    mean_intensity = roi.mean()
    std_intensity = roi.std()
    
    dark_threshold = mean_intensity - 1.5 * std_intensity
    dark_mask = (roi < dark_threshold).astype(np.uint8)
    
    #morphology to clean up
    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    
    dark_ratio = dark_mask.mean()
    
    #find contours of dark regions
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_dark_regions = sum(1 for c in contours if cv2.contourArea(c) > 50)
    
    return dark_ratio, large_dark_regions

#=== MOTION ANALYSIS (OIS-aware) ===

def compute_motion_features(frames, window=5):
    """compute motion features that survive OIS stabilization"""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = grays[0].shape
    roi_start = h // 3
    
    motion_features = []
    
    for i in range(len(grays) - 1):
        prev = grays[i].astype(float)
        curr = grays[i + 1].astype(float)
        
        #frame difference
        diff = np.abs(curr - prev)
        
        #focus on bottom region (more affected by bumps)
        bottom_diff = diff[2*h//3:, :]
        middle_diff = diff[h//3:2*h//3, :]
        
        #intensity changes (bumps cause exposure/brightness flicker)
        intensity_change = np.abs(curr.mean() - prev.mean())
        
        #local variance change (texture appears/disappears during bump)
        prev_var = cv2.Laplacian(prev[roi_start:], cv2.CV_64F).var()
        curr_var = cv2.Laplacian(curr[roi_start:], cv2.CV_64F).var()
        variance_change = np.abs(curr_var - prev_var)
        
        #edge density change
        prev_edges = cv2.Canny(grays[i][roi_start:], 50, 150)
        curr_edges = cv2.Canny(grays[i+1][roi_start:], 50, 150)
        edge_change = np.abs(curr_edges.mean() - prev_edges.mean())
        
        motion_features.append({
            'frame_diff_mean': diff.mean(),
            'frame_diff_bottom': bottom_diff.mean(),
            'frame_diff_middle': middle_diff.mean(),
            'intensity_change': intensity_change,
            'variance_change': variance_change,
            'edge_change': edge_change,
        })
    
    return motion_features

def detect_oscillation_pattern(signal_data, fps=20):
    """detect bump-like oscillation patterns (up-down-up motion)"""
    if len(signal_data) < 10:
        return np.zeros(len(signal_data))
    
    #smooth to reduce noise
    smoothed = gaussian_filter1d(signal_data, sigma=1)
    
    #find zero crossings of derivative (peaks and valleys)
    derivative = np.diff(smoothed)
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    
    #oscillation score: rapid zero crossings = bump
    oscillation_score = np.zeros(len(signal_data))
    
    window = 5  #frames
    for i in range(len(signal_data)):
        start = max(0, i - window)
        end = min(len(signal_data), i + window)
        crossings_in_window = np.sum((zero_crossings >= start) & (zero_crossings < end))
        oscillation_score[i] = crossings_in_window / window
    
    return oscillation_score

def frequency_analysis(signal_data, fps=20):
    """detect bump frequency signature (bumps have characteristic frequency)"""
    if len(signal_data) < 20:
        return np.zeros(len(signal_data))
    
    window_size = 20
    bump_freq_scores = np.zeros(len(signal_data))
    
    #bump frequency range: 2-8 Hz (typical scooter bump duration)
    bump_freq_low = 2
    bump_freq_high = 8
    
    for i in range(window_size, len(signal_data)):
        window = signal_data[i-window_size:i]
        window = window - window.mean()  #remove DC
        
        fft_vals = np.abs(fft(window))
        freqs = fftfreq(window_size, 1/fps)
        
        #energy in bump frequency range
        bump_mask = (np.abs(freqs) >= bump_freq_low) & (np.abs(freqs) <= bump_freq_high)
        total_energy = np.sum(fft_vals ** 2) + 1e-10
        bump_energy = np.sum(fft_vals[bump_mask] ** 2)
        
        bump_freq_scores[i] = bump_energy / total_energy
    
    return bump_freq_scores

#=== MULTI-MODAL FUSION ===

def extract_all_features(frames):
    """extract comprehensive features for bump detection"""
    print("extracting visual and motion features...")
    
    n_frames = len(frames)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = grays[0].shape
    roi_start = h // 3
    
    #visual features per frame
    visual_features = []
    for i, gray in enumerate(grays):
        h_score, h_strength = detect_horizontal_lines(gray, roi_start)
        tex_var, roughness, bottom_rough = detect_texture_roughness(gray, roi_start)
        shadow_score = detect_shadow_lines(gray, roi_start)
        dark_ratio, dark_regions = detect_dark_regions(gray, roi_start)
        
        visual_features.append({
            'horizontal_line_score': h_score,
            'horizontal_strength': h_strength,
            'texture_variance': tex_var,
            'roughness': roughness,
            'bottom_roughness': bottom_rough,
            'shadow_score': shadow_score,
            'dark_ratio': dark_ratio,
            'dark_regions': dark_regions,
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  visual features: {i + 1}/{n_frames}")
    
    #motion features
    motion_features = compute_motion_features(frames)
    
    #combine into arrays
    feature_names = list(visual_features[0].keys()) + list(motion_features[0].keys())
    
    #align lengths (motion has n-1 elements)
    all_features = []
    for i in range(len(motion_features)):
        combined = {**visual_features[i], **motion_features[i]}
        all_features.append(combined)
    
    #convert to arrays
    feature_arrays = {k: np.array([f[k] for f in all_features]) for k in feature_names}
    
    return feature_arrays, feature_names

def compute_bump_scores(feature_arrays, fps=20):
    """compute multi-modal bump scores"""
    n = len(feature_arrays['frame_diff_mean'])
    
    #1. visual defect score
    visual_score = (
        feature_arrays['horizontal_line_score'] * 2 +
        feature_arrays['horizontal_strength'] / 50 +
        feature_arrays['bottom_roughness'] / 20 +
        feature_arrays['shadow_score'] * 10 +
        feature_arrays['dark_ratio'] * 5
    )
    visual_score = (visual_score - visual_score.mean()) / (visual_score.std() + 1e-6)
    
    #2. motion score (OIS-resistant)
    motion_score = (
        feature_arrays['frame_diff_bottom'] * 2 +
        feature_arrays['variance_change'] / 100 +
        feature_arrays['edge_change'] * 5
    )
    motion_score = (motion_score - motion_score.mean()) / (motion_score.std() + 1e-6)
    
    #3. oscillation score
    bottom_diff = feature_arrays['frame_diff_bottom']
    oscillation = detect_oscillation_pattern(bottom_diff, fps)
    
    #4. frequency score
    freq_score = frequency_analysis(bottom_diff, fps)
    
    #5. temporal derivative (sudden changes)
    motion_derivative = np.abs(np.diff(motion_score, prepend=motion_score[0]))
    motion_derivative = gaussian_filter1d(motion_derivative, sigma=1)
    
    #combined score with weights
    combined_score = (
        visual_score * 0.25 +           #visual defects
        motion_score * 0.25 +           #motion intensity
        oscillation * 2.0 +             #oscillation pattern
        freq_score * 1.5 +              #frequency signature
        motion_derivative * 1.0         #sudden changes
    )
    
    #smooth slightly
    combined_score = gaussian_filter1d(combined_score, sigma=1)
    
    return {
        'visual': visual_score,
        'motion': motion_score,
        'oscillation': oscillation,
        'frequency': freq_score,
        'derivative': motion_derivative,
        'combined': combined_score
    }

def detect_bumps_robust(scores, threshold_percentile=90, min_distance=8):
    """detect bumps using combined scores with temporal consistency"""
    combined = scores['combined']
    
    #adaptive threshold
    threshold = np.percentile(combined, threshold_percentile)
    
    #find peaks
    peaks, properties = signal.find_peaks(
        combined,
        height=threshold,
        distance=min_distance,
        prominence=0.5  #require some prominence
    )
    
    #validate with temporal consistency
    #a real bump should show in multiple signals
    valid_peaks = []
    valid_intensities = []
    
    for peak in peaks:
        #check if multiple signals agree
        window = 3
        start = max(0, peak - window)
        end = min(len(combined), peak + window + 1)
        
        visual_active = np.max(scores['visual'][start:end]) > 0.5
        motion_active = np.max(scores['motion'][start:end]) > 0.5
        freq_active = np.max(scores['frequency'][start:end]) > 0.1
        
        #require at least 2 signals to agree
        agreement = int(visual_active) + int(motion_active) + int(freq_active)
        
        if agreement >= 1:  #at least one signal confirms
            valid_peaks.append(peak)
            valid_intensities.append(combined[peak])
    
    return np.array(valid_peaks), np.array(valid_intensities), combined, threshold

def filter_false_positives(bump_frames, bump_intensities, scores, min_intensity=1.0):
    """filter out likely false positives"""
    filtered_frames = []
    filtered_intensities = []
    
    for bf, bi in zip(bump_frames, bump_intensities):
        #check for consistent motion pattern (not just noise)
        window = 5
        start = max(0, bf - window)
        end = min(len(scores['motion']), bf + window + 1)
        
        local_motion = scores['motion'][start:end]
        motion_std = local_motion.std()
        
        #real bumps have motion variability
        if motion_std > 0.3 or bi > min_intensity:
            filtered_frames.append(bf)
            filtered_intensities.append(bi)
    
    return np.array(filtered_frames), np.array(filtered_intensities)

#=== MAIN ===

def main():
    #find video
    video_files = list(DATA_DIR.glob("*.mp4")) + list(DATA_DIR.glob("*.mov"))
    if not video_files:
        print("no videos found!")
        return
    
    video_path = video_files[0]
    print(f"processing: {video_path.name}\n")
    
    #load video
    print("loading video...")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) % 2000 == 0:
            print(f"  loaded {len(frames)}/{total}")
    cap.release()
    frames = np.array(frames)
    print(f"  loaded {len(frames)} frames\n")
    
    #extract features
    feature_arrays, feature_names = extract_all_features(frames)
    
    #compute bump scores
    print("\ncomputing bump scores...")
    scores = compute_bump_scores(feature_arrays, FPS)
    
    #detect bumps
    print("detecting bumps...")
    bump_frames, bump_intensities, combined_signal, threshold = detect_bumps_robust(
        scores, threshold_percentile=90, min_distance=8
    )
    print(f"  initial detection: {len(bump_frames)} bumps")
    
    #filter false positives
    bump_frames, bump_intensities = filter_false_positives(
        bump_frames, bump_intensities, scores, min_intensity=1.0
    )
    print(f"  after filtering: {len(bump_frames)} bumps")
    
    #save results
    detection_results = {
        'bump_frames': bump_frames,
        'bump_intensities': bump_intensities,
        'bump_threshold': threshold,
        'scores': scores,
        'feature_arrays': feature_arrays,
        'fps': FPS
    }
    
    with open(OUTPUT_DIR / 'bump_detection_robust.pkl', 'wb') as f:
        pickle.dump(detection_results, f)
    print(f"\nsaved to {OUTPUT_DIR / 'bump_detection_robust.pkl'}")
    
    #also update the standard detection file for compatibility
    with open(OUTPUT_DIR / 'bump_detection.pkl', 'wb') as f:
        pickle.dump({
            'bump_frames': bump_frames,
            'bump_intensities': bump_intensities,
            'bump_threshold': threshold,
            'bump_labels': np.zeros(len(frames)),  #placeholder
            'flow_features': {'avg_vy': scores['motion']},  #compatibility
            'fps': FPS
        }, f)
    
    #plot results
    print("\ngenerating plots...")
    time = np.arange(len(scores['combined'])) / FPS
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    
    axes[0].plot(time, scores['visual'], 'b-', alpha=0.7, label='visual defects')
    axes[0].set_ylabel('visual score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, scores['motion'], 'r-', alpha=0.7, label='motion')
    axes[1].set_ylabel('motion score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time, scores['oscillation'], 'g-', alpha=0.7, label='oscillation')
    axes[2].set_ylabel('oscillation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(time, scores['frequency'], 'm-', alpha=0.7, label='freq signature')
    axes[3].set_ylabel('frequency')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    axes[4].plot(time, scores['combined'], 'k-', alpha=0.7, label='combined')
    axes[4].axhline(y=threshold, color='r', linestyle='--', label='threshold')
    bump_times = bump_frames / FPS
    axes[4].scatter(bump_times, bump_intensities, c='red', s=50, marker='v', label='bumps', zorder=5)
    axes[4].set_ylabel('combined')
    axes[4].set_xlabel('time (s)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    plt.suptitle('robust multi-modal bump detection')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bump_detection_robust_plot.png', dpi=150)
    print(f"  saved {OUTPUT_DIR / 'bump_detection_robust_plot.png'}")
    
    #histogram of bump intensities
    plt.figure(figsize=(10, 5))
    plt.hist(bump_intensities, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('bump intensity')
    plt.ylabel('count')
    plt.title(f'distribution of {len(bump_frames)} detected bumps')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bump_intensity_histogram.png', dpi=150)
    print(f"  saved {OUTPUT_DIR / 'bump_intensity_histogram.png'}")
    
    #print summary
    print("\n" + "="*60)
    print("ROBUST BUMP DETECTION SUMMARY")
    print("="*60)
    print(f"video: {video_path.name}")
    print(f"duration: {len(frames)/FPS:.1f} seconds")
    print(f"bumps detected: {len(bump_frames)}")
    print(f"threshold: {threshold:.3f}")
    print(f"\nbump intensity stats:")
    print(f"  min: {bump_intensities.min():.2f}")
    print(f"  max: {bump_intensities.max():.2f}")
    print(f"  mean: {bump_intensities.mean():.2f}")
    print(f"  median: {np.median(bump_intensities):.2f}")
    
    #show top 10 strongest bumps
    print(f"\ntop 10 strongest bumps:")
    sorted_idx = np.argsort(bump_intensities)[::-1]
    for i in range(min(10, len(bump_frames))):
        idx = sorted_idx[i]
        frame = bump_frames[idx]
        intensity = bump_intensities[idx]
        time_s = frame / FPS
        print(f"  {i+1}. frame {frame} (t={time_s:.1f}s), intensity={intensity:.2f}")

if __name__ == "__main__":
    main()




