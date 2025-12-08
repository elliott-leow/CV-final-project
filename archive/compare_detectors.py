#compare optical flow vs robust bump detection
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("output")
FPS = 20

def load_detections():
    """load both detection results"""
    #try loading optical flow results (from original run)
    try:
        with open(OUTPUT_DIR / 'bump_detection_original.pkl', 'rb') as f:
            optical_flow = pickle.load(f)
    except:
        optical_flow = None
    
    #load robust detection
    with open(OUTPUT_DIR / 'bump_detection_robust.pkl', 'rb') as f:
        robust = pickle.load(f)
    
    return optical_flow, robust

def compare():
    optical_flow, robust = load_detections()
    
    robust_frames = set(robust['bump_frames'])
    robust_intensities = robust['bump_intensities']
    scores = robust['scores']
    
    print("="*60)
    print("ROBUST BUMP DETECTOR ANALYSIS")
    print("="*60)
    
    print(f"\ndetected {len(robust_frames)} bumps")
    
    #analyze score contributions
    print("\nscore signal analysis:")
    for name, signal in scores.items():
        print(f"  {name}: mean={signal.mean():.3f}, std={signal.std():.3f}, max={signal.max():.3f}")
    
    #intensity distribution
    print(f"\nbump intensity distribution:")
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(robust_intensities, p)
        print(f"  {p}th percentile: {val:.2f}")
    
    #temporal distribution
    frame_times = np.array(list(robust_frames)) / FPS
    print(f"\ntemporal distribution:")
    print(f"  first bump: {frame_times.min():.1f}s")
    print(f"  last bump: {frame_times.max():.1f}s")
    
    #inter-bump intervals
    sorted_frames = np.sort(list(robust_frames))
    intervals = np.diff(sorted_frames) / FPS
    print(f"\ninter-bump intervals:")
    print(f"  min: {intervals.min():.2f}s")
    print(f"  max: {intervals.max():.2f}s")
    print(f"  mean: {intervals.mean():.2f}s")
    print(f"  median: {np.median(intervals):.2f}s")
    
    #create visualization
    time = np.arange(len(scores['combined'])) / FPS
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    #combined signal with bumps
    axes[0].plot(time, scores['combined'], 'b-', alpha=0.7, linewidth=0.5)
    axes[0].axhline(y=robust['bump_threshold'], color='r', linestyle='--', alpha=0.5)
    bump_times = np.array(list(robust_frames)) / FPS
    axes[0].scatter(bump_times, [scores['combined'][f] for f in robust_frames if f < len(scores['combined'])], 
                    c='red', s=20, marker='v', zorder=5)
    axes[0].set_ylabel('combined score')
    axes[0].set_title('robust bump detection')
    axes[0].grid(True, alpha=0.3)
    
    #score components
    axes[1].plot(time, scores['visual'], 'g-', alpha=0.5, label='visual', linewidth=0.5)
    axes[1].plot(time, scores['motion'], 'b-', alpha=0.5, label='motion', linewidth=0.5)
    axes[1].plot(time, scores['oscillation'] * 2, 'r-', alpha=0.5, label='oscillation', linewidth=0.5)
    axes[1].set_ylabel('component scores')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    #bump density over time
    window_sec = 30
    window_frames = window_sec * FPS
    bump_density = np.zeros(len(scores['combined']))
    for f in robust_frames:
        if f < len(bump_density):
            start = max(0, f - window_frames//2)
            end = min(len(bump_density), f + window_frames//2)
            bump_density[start:end] += 1
    bump_density = bump_density / window_sec  #bumps per second
    
    axes[2].fill_between(time, 0, bump_density, alpha=0.5, color='orange')
    axes[2].set_ylabel('bump density\n(per second)')
    axes[2].set_xlabel('time (seconds)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detector_comparison.png', dpi=150)
    print(f"\nsaved {OUTPUT_DIR / 'detector_comparison.png'}")
    
    #print high-bump-density regions
    print("\nhigh bump density regions:")
    threshold_density = np.percentile(bump_density, 95)
    high_density_starts = []
    in_high = False
    for i, d in enumerate(bump_density):
        if d > threshold_density and not in_high:
            in_high = True
            high_density_starts.append(i)
        elif d <= threshold_density:
            in_high = False
    
    for start in high_density_starts[:10]:
        time_s = start / FPS
        density = bump_density[start]
        print(f"  t={time_s:.1f}s: density={density:.2f} bumps/sec")

if __name__ == "__main__":
    compare()




