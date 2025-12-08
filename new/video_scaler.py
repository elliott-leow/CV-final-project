#video scaling pipeline with edge preservation using ffmpeg
#downscales to 240p at 15fps while maintaining sharp edges

import cv2
import numpy as np
import subprocess
import os
import shutil

import config


def check_ffmpeg():
    """check if ffmpeg is available"""
    return shutil.which('ffmpeg') is not None


def get_video_info(input_path):
    """get video info using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        '-of', 'csv=p=0',
        input_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        parts = result.stdout.strip().split(',')
        width = int(parts[0])
        height = int(parts[1])
        fps_parts = parts[2].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        nb_frames = int(parts[3]) if len(parts) > 3 and parts[3] != 'N/A' else None
        return {'width': width, 'height': height, 'fps': fps, 'nb_frames': nb_frames}
    except:
        return None


def scale_video_ffmpeg(input_path, output_path, target_width, target_height, target_fps):
    """scale video using ffmpeg with high quality settings"""
    filter_chain = (
        f"fps={target_fps},"
        f"scale={target_width}:{target_height}:flags=lanczos,"
        f"unsharp=5:5:0.5:5:5:0.0"
    )
    
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', filter_chain,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-an',
        output_path
    ]
    
    print(f"running ffmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return False
    
    return True


def scale_video_ffmpeg_gpu(input_path, output_path, target_width, target_height, target_fps):
    """scale video using ffmpeg with nvidia gpu acceleration"""
    filter_chain = (
        f"fps={target_fps},"
        f"scale={target_width}:{target_height}:flags=lanczos,"
        f"unsharp=5:5:0.5:5:5:0.0"
    )
    
    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-i', input_path,
        '-vf', filter_chain,
        '-c:v', 'h264_nvenc',
        '-preset', 'p4',
        '-cq', '18',
        '-an',
        output_path
    ]
    
    print("attempting gpu encoding with nvenc...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("gpu encoding successful")
        return True
    
    print("nvenc failed, trying intel qsv...")
    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'qsv',
        '-i', input_path,
        '-vf', filter_chain,
        '-c:v', 'h264_qsv',
        '-preset', 'fast',
        '-global_quality', '18',
        '-an',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("intel qsv encoding successful")
        return True
    
    print("gpu encoding failed, falling back to cpu...")
    return scale_video_ffmpeg(input_path, output_path, target_width, target_height, target_fps)


def scale_video(input_path, output_path=None, 
                target_width=None, target_height=None, target_fps=None,
                use_gpu=True):
    """
    main function to scale video with edge preservation
    returns frame count and output path (does NOT load frames into memory)
    """
    target_width = target_width or config.TARGET_WIDTH
    target_height = target_height or config.TARGET_HEIGHT
    target_fps = target_fps or config.TARGET_FPS
    
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_scaled.mp4"
    
    info = get_video_info(input_path)
    if info:
        print(f"input: {info['width']}x{info['height']} @ {info['fps']:.2f}fps")
    print(f"output: {target_width}x{target_height} @ {target_fps}fps")
    
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg not found")
    
    if use_gpu:
        success = scale_video_ffmpeg_gpu(input_path, output_path, target_width, target_height, target_fps)
    else:
        success = scale_video_ffmpeg(input_path, output_path, target_width, target_height, target_fps)
    
    if not success:
        raise RuntimeError("video scaling failed")
    
    print(f"saved scaled video to {output_path}")
    
    #get frame count without loading
    frame_count = get_frame_count(output_path)
    print(f"scaled video has {frame_count} frames")
    
    return frame_count, output_path


def get_frame_count(video_path):
    """get frame count without loading video"""
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def load_video_chunk(video_path, start_frame, num_frames):
    """load a chunk of frames from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    
    cap.release()
    return np.array(frames) if frames else None


def iter_video_frames(video_path, batch_size=100):
    """generator that yields batches of frames"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    batch = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch.append(rgb)
        frame_idx += 1
        
        if len(batch) >= batch_size:
            yield np.array(batch), frame_idx - len(batch)
            batch = []
    
    if batch:
        yield np.array(batch), frame_idx - len(batch)
    
    cap.release()


def load_scaled_video(video_path, max_frames=None):
    """load video frames (with optional limit)"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    
    cap.release()
    print(f"loaded {len(frames)} frames from {video_path}")
    return np.array(frames)


if __name__ == "__main__":
    video_path = os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4")
    output_path = os.path.join(config.OUTPUT_DIR, "scaled_video.mp4")
    
    if os.path.exists(video_path):
        frame_count, out_path = scale_video(video_path, output_path)
        print(f"processed video with {frame_count} frames")
    else:
        print(f"video not found: {video_path}")
