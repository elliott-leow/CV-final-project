#audio-based ground truth generation for bump detection
#uses high frequency energy (>4000Hz) to identify bumps

import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks
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


def highpass_filter(audio, sr, cutoff_freq):
    """apply highpass filter to isolate high frequency bump sounds"""
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))
    
    b, a = butter(4, normalized_cutoff, btype='high')
    filtered = filtfilt(b, a, audio)
    return filtered


def compute_energy_envelope(audio, sr, window_sec=0.05):
    """compute rolling energy envelope of audio signal"""
    window_size = int(sr * window_sec)
    
    energy = audio ** 2
    kernel = np.ones(window_size) / window_size
    envelope = np.convolve(energy, kernel, mode='same')
    
    return envelope


def detect_bumps_from_audio(audio, sr, video_fps, 
                            min_freq=4000,
                            threshold_percentile=90,
                            min_distance_sec=0.15,
                            bump_duration_sec=0.15):
    """
    detect bumps using audio energy in high frequency band (>4000Hz)
    bumps typically have 0.1-0.2s of spectral energy
    """
    print(f"  using highpass filter: >{min_freq}Hz")
    print(f"  expected bump duration: ~{bump_duration_sec}s")
    
    #step 1: highpass filter to get high frequency content
    filtered = highpass_filter(audio, sr, min_freq)
    
    #step 2: compute energy envelope with window matching bump duration
    window_sec = bump_duration_sec / 2  #half the bump duration for responsiveness
    envelope = compute_energy_envelope(filtered, sr, window_sec=window_sec)
    
    #step 3: threshold based on percentile
    threshold = np.percentile(envelope, threshold_percentile)
    
    #step 4: find peaks above threshold
    min_distance_samples = int(sr * min_distance_sec)
    peaks, properties = find_peaks(envelope, 
                                   height=threshold,
                                   distance=min_distance_samples,
                                   prominence=threshold * 0.1)
    
    #step 5: convert sample indices to frame indices
    samples_per_frame = sr / video_fps
    bump_frames = (peaks / samples_per_frame).astype(int)
    
    #step 6: get peak heights for confidence scores
    peak_heights = properties['peak_heights']
    confidence = (peak_heights - threshold) / (np.max(envelope) - threshold + 1e-8)
    confidence = np.clip(confidence, 0, 1)
    
    return bump_frames, confidence, envelope, threshold


def generate_ground_truth(video_path, save_path=None):
    """generate ground truth bump labels from video audio"""
    print(f"extracting audio from {video_path}...")
    audio, sr, video_fps, duration = extract_audio_from_video(video_path)
    
    print(f"video fps: {video_fps:.2f}, duration: {duration:.2f}s")
    print(f"audio sample rate: {sr}, length: {len(audio)} samples")
    
    print("detecting bumps from high-frequency audio...")
    bump_frames, confidence, envelope, threshold = detect_bumps_from_audio(
        audio, sr, video_fps
    )
    
    print(f"detected {len(bump_frames)} bumps")
    
    bump_times = bump_frames / video_fps
    
    ground_truth = {
        'bump_frames': bump_frames,
        'bump_times': bump_times,
        'confidence': confidence,
        'video_fps': video_fps,
        'duration': duration,
        'total_frames': int(duration * video_fps),
        'envelope': envelope,
        'threshold': threshold,
        'audio_sr': sr
    }
    
    if save_path:
        np.save(save_path, ground_truth)
        print(f"saved ground truth to {save_path}")
    
    return ground_truth


def visualize_ground_truth(ground_truth, save_path=None):
    """visualize audio envelope and detected bumps"""
    import matplotlib.pyplot as plt
    
    envelope = ground_truth['envelope']
    threshold = ground_truth['threshold']
    sr = ground_truth['audio_sr']
    bump_times = ground_truth['bump_times']
    
    time_axis = np.arange(len(envelope)) / sr
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time_axis, envelope, 'b-', alpha=0.7, label='high-freq energy (>4kHz)')
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'threshold')
    
    for t in bump_times:
        ax.axvline(x=t, color='g', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('time (s)')
    ax.set_ylabel('energy')
    ax.set_title(f'bump detection from high-freq audio - {len(bump_times)} bumps found')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved visualization to {save_path}")
    
    plt.close(fig)
    return fig


if __name__ == "__main__":
    video_path = os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4")
    
    if os.path.exists(video_path):
        gt = generate_ground_truth(
            video_path, 
            save_path=os.path.join(config.OUTPUT_DIR, "ground_truth.npy")
        )
        visualize_ground_truth(
            gt, 
            save_path=os.path.join(config.OUTPUT_DIR, "ground_truth_viz.png")
        )
    else:
        print(f"video not found: {video_path}")
