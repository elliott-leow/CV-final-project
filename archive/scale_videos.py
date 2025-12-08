#video scaling script - converts videos to 20fps and 240p
import cv2
import os
from pathlib import Path

def scale_video(input_path, output_path, target_fps=20, target_height=240):
    """scale video to target fps and resolution (240p = 240 height, width proportional)"""
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"error: could not open {input_path}")
        return False
    
    #get original properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #calculate new dimensions (maintain aspect ratio)
    scale = target_height / orig_height
    target_width = int(orig_width * scale)
    #ensure even dimensions for codec compatibility
    target_width = target_width if target_width % 2 == 0 else target_width + 1
    
    print(f"processing: {input_path.name}")
    print(f"  original: {orig_width}x{orig_height} @ {orig_fps:.1f}fps, {total_frames} frames")
    print(f"  target: {target_width}x{target_height} @ {target_fps}fps")
    
    #calculate frame skip ratio
    frame_interval = orig_fps / target_fps
    
    #setup output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (target_width, target_height))
    
    frame_idx = 0
    next_frame_to_grab = 0
    written = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #only keep frames at the target fps rate
        if frame_idx >= next_frame_to_grab:
            #resize frame
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            out.write(resized)
            written += 1
            next_frame_to_grab += frame_interval
        
        frame_idx += 1
        
        #progress indicator
        if frame_idx % 100 == 0:
            print(f"  progress: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
    
    cap.release()
    out.release()
    print(f"  output: {written} frames written to {output_path.name}")
    return True

def main():
    data_dir = Path("data")
    output_dir = Path("data-scaled")
    
    #create output directory
    output_dir.mkdir(exist_ok=True)
    
    #supported video extensions
    video_exts = {'.mov', '.mp4', '.avi', '.mkv', '.webm', '.m4v'}
    
    #find all videos
    videos = [f for f in data_dir.iterdir() if f.suffix.lower() in video_exts]
    
    if not videos:
        print(f"no videos found in {data_dir}")
        return
    
    print(f"found {len(videos)} video(s) to process\n")
    
    for video_path in videos:
        output_name = video_path.stem + "_scaled.mp4"
        output_path = output_dir / output_name
        scale_video(video_path, output_path)
        print()
    
    print("done!")

if __name__ == "__main__":
    main()


