#steps 1-2: audio-based bump candidate generation with gaussian priors
#uses bandpass filtering to identify approximate bump times

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import maximum_filter1d
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip
import os

import config


def extract_audio_from_video(video_path, output_path=None):
    """extract audio from video file and return as numpy array"""
    clip = VideoFileClip(video_path)
    
    if output_path:
        clip.audio.write_audiofile(output_path, fps=config.AUDIO_SAMPLE_RATE)
    
    audio = clip.audio.to_soundarray(fps=config.AUDIO_SAMPLE_RATE)
    
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
    
    fps = clip.fps
    duration = clip.duration
    clip.close()
    
    return audio, config.AUDIO_SAMPLE_RATE, fps, duration


def bandpass_filter(audio, sr, low_freq, high_freq):
    """apply bandpass filter to isolate bump frequency band"""
    nyquist = sr / 2
    low = max(0.01, low_freq / nyquist)
    high = min(0.99, high_freq / nyquist)
    
    if low >= high:
        low = high - 0.01
    
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)
    return filtered


def compute_energy_envelope(audio, sr, window_sec=0.05):
    """compute rolling energy envelope of audio signal"""
    window_size = int(sr * window_sec)
    
    energy = audio ** 2
    kernel = np.ones(window_size) / window_size
    envelope = np.convolve(energy, kernel, mode='same')
    
    return envelope


def detect_audio_bump_candidates(audio, sr, video_fps,
                                  min_freq=None, max_freq=None,
                                  threshold_percentile=None,
                                  min_distance_sec=None):
    """
    step 1: detect bump candidates using audio energy in frequency band
    returns set of approximate bump times L_k (in frames)
    """
    min_freq = min_freq or config.AUDIO_MIN_FREQ_HZ
    max_freq = max_freq or config.AUDIO_MAX_FREQ_HZ
    threshold_percentile = threshold_percentile or config.ENERGY_THRESHOLD_PERCENTILE
    min_distance_sec = min_distance_sec or config.MIN_BUMP_DISTANCE_SEC
    
    print(f"  bandpass filter: {min_freq}Hz - {max_freq}Hz")
    
    #bandpass filter
    filtered = bandpass_filter(audio, sr, min_freq, max_freq)
    
    #compute energy envelope
    envelope = compute_energy_envelope(filtered, sr, window_sec=0.05)
    
    #threshold
    threshold = np.percentile(envelope, threshold_percentile)
    
    #find peaks
    min_distance_samples = int(sr * min_distance_sec)
    peaks, properties = find_peaks(envelope,
                                   height=threshold,
                                   distance=min_distance_samples,
                                   prominence=threshold * 0.1)
    
    #merge nearby detections (closer than min_distance)
    if len(peaks) > 1:
        merged_peaks = [peaks[0]]
        for p in peaks[1:]:
            if (p - merged_peaks[-1]) > min_distance_samples:
                merged_peaks.append(p)
            else:
                #keep the higher energy one
                if envelope[p] > envelope[merged_peaks[-1]]:
                    merged_peaks[-1] = p
        peaks = np.array(merged_peaks)
    
    #convert to frame indices
    samples_per_frame = sr / video_fps
    bump_frames = (peaks / samples_per_frame).astype(int)
    
    #get confidence scores
    peak_heights = envelope[peaks]
    confidence = (peak_heights - threshold) / (np.max(envelope) - threshold + 1e-8)
    confidence = np.clip(confidence, 0, 1)
    
    print(f"  detected {len(bump_frames)} audio bump candidates")
    
    return bump_frames, confidence, envelope, threshold


def create_candidate_windows(bump_frames, total_frames, radius=None, sigma=None):
    """
    step 2: create temporal windows around each audio candidate
    for each L_k, define T_k = {L_k - r, ..., L_k + r}
    with gaussian prior over offsets
    
    returns:
        candidates: list of dicts with:
            - audio_frame: the audio-detected frame L_k
            - window: array of candidate true bump frames T_k
            - prior: normalized gaussian prior pi_k(t) over window
    """
    radius = radius or config.CANDIDATE_RADIUS
    sigma = sigma or config.GAUSSIAN_SIGMA
    
    candidates = []
    
    for L_k in bump_frames:
        #define window T_k
        t_min = max(0, L_k - radius)
        t_max = min(total_frames - 1, L_k + radius)
        window = np.arange(t_min, t_max + 1)
        
        #compute gaussian prior pi_k(t)
        offsets = window - L_k
        prior = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
        prior = prior / prior.sum()  #normalize
        
        candidates.append({
            'audio_frame': L_k,
            'window': window,
            'prior': prior
        })
    
    return candidates


def get_all_candidate_frames(candidates):
    """get all frames that are candidates for some bump"""
    all_frames = set()
    for c in candidates:
        all_frames.update(c['window'])
    return sorted(list(all_frames))


def generate_bump_candidates(video_path, target_fps=None, save_path=None):
    """
    steps 1-2: generate audio bump candidates with temporal windows
    
    returns dict with:
        - candidates: list of candidate dicts (audio_frame, window, prior)
        - bump_frames: audio-detected bump frames (L_k)
        - confidence: confidence scores for audio detections
        - envelope: audio energy envelope
        - metadata: video info
    """
    target_fps = target_fps or config.TARGET_FPS
    
    print(f"extracting audio from {video_path}...")
    audio, sr, video_fps, duration = extract_audio_from_video(video_path)
    
    print(f"video fps: {video_fps:.2f}, target fps: {target_fps}")
    print(f"audio sample rate: {sr}, length: {len(audio)} samples")
    
    #step 1: detect audio bump candidates
    print("step 1: detecting audio bump candidates...")
    bump_frames_original, confidence, envelope, threshold = detect_audio_bump_candidates(
        audio, sr, video_fps
    )
    
    #convert to target fps
    fps_ratio = target_fps / video_fps
    bump_frames = (bump_frames_original * fps_ratio).astype(int)
    total_frames = int(duration * target_fps)
    
    #step 2: create candidate windows with priors
    print("step 2: creating candidate windows with gaussian priors...")
    print(f"  window radius: {config.CANDIDATE_RADIUS} frames")
    print(f"  gaussian sigma: {config.GAUSSIAN_SIGMA}")
    candidates = create_candidate_windows(bump_frames, total_frames)
    
    result = {
        'candidates': candidates,
        'bump_frames': bump_frames,
        'confidence': confidence,
        'envelope': envelope,
        'threshold': threshold,
        'video_fps': video_fps,
        'target_fps': target_fps,
        'duration': duration,
        'total_frames': total_frames,
        'audio_sr': sr
    }
    
    if save_path:
        np.save(save_path, result)
        print(f"saved bump candidates to {save_path}")
    
    return result


def visualize_bump_candidates(result, save_path=None):
    """visualize audio envelope and bump candidates with windows"""
    import matplotlib.pyplot as plt
    
    envelope = result['envelope']
    threshold = result['threshold']
    sr = result['audio_sr']
    candidates = result['candidates']
    target_fps = result['target_fps']
    
    time_axis = np.arange(len(envelope)) / sr
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    #plot 1: energy envelope with threshold
    axes[0].plot(time_axis, envelope, 'b-', alpha=0.7, label='bandpass energy')
    axes[0].axhline(y=threshold, color='r', linestyle='--', label='threshold')
    
    for c in candidates:
        t = c['audio_frame'] / target_fps
        axes[0].axvline(x=t, color='g', alpha=0.5, linewidth=2)
    
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('energy')
    axes[0].set_title(f'audio bump detection - {len(candidates)} candidates')
    axes[0].legend()
    
    #plot 2: candidate windows with priors
    for i, c in enumerate(candidates[:20]):  #limit to first 20
        window_times = c['window'] / target_fps
        prior = c['prior']
        audio_time = c['audio_frame'] / target_fps
        
        axes[1].fill_between(window_times, 0, prior, alpha=0.3)
        axes[1].axvline(x=audio_time, color='g', linestyle='--', alpha=0.5)
    
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('prior probability')
    axes[1].set_title('candidate windows with gaussian priors (first 20)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved visualization to {save_path}")
    
    plt.close(fig)
    return fig


#backwards compatible alias
def generate_ground_truth(video_path, save_path=None):
    """alias for generate_bump_candidates for backwards compatibility"""
    return generate_bump_candidates(video_path, save_path=save_path)


if __name__ == "__main__":
    video_path = os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4")
    
    if os.path.exists(video_path):
        result = generate_bump_candidates(
            video_path,
            save_path=os.path.join(config.OUTPUT_DIR, "bump_candidates.npy")
        )
        visualize_bump_candidates(
            result,
            save_path=os.path.join(config.OUTPUT_DIR, "bump_candidates_viz.png")
        )
    else:
        print(f"video not found: {video_path}")
