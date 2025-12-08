#edge detection pipeline - passes all canny edges to model without filtering
#the model will learn which edges are relevant for bump detection

import cv2
import numpy as np
from tqdm import tqdm
import os

import config


def canny_edge_detection(frame, low_thresh=None, high_thresh=None):
    """apply canny edge detection to a single frame"""
    low_thresh = low_thresh or config.CANNY_LOW_THRESHOLD
    high_thresh = high_thresh or config.CANNY_HIGH_THRESHOLD
    
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    return edges


def process_frame_edges(frame):
    """process single frame - just canny edges, no filtering"""
    edges = canny_edge_detection(frame)
    return edges


def extract_edges_batch(frames):
    """extract canny edges from all frames without any filtering"""
    edges = []
    for frame in tqdm(frames, desc="extracting canny edges"):
        edge = canny_edge_detection(frame)
        edges.append(edge)
    return np.array(edges)


def process_edges(frames):
    """
    edge processing - pass all canny edges without filtering
    let the model learn which edges are relevant
    """
    print("extracting all canny edges (no filtering)...")
    edges = extract_edges_batch(frames)
    
    total_edge_pixels = np.sum(edges > 0)
    print(f"  extracted {len(edges)} edge frames")
    print(f"  total edge pixels: {total_edge_pixels:,}")
    
    return {
        'raw_edges': edges,
        'filtered': edges  #same as raw, no filtering
    }


def create_feature_video(frames, edges, output_path):
    """create video showing the edge features"""
    T, H, W = edges.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, (W, H))
    
    for t in tqdm(range(T), desc="creating feature video"):
        edge_frame = cv2.cvtColor(edges[t], cv2.COLOR_GRAY2BGR)
        out.write(edge_frame)
    
    out.release()
    print(f"saved feature video to {output_path}")


def create_overlay_video(frames, edges, output_path):
    """create video with edge overlay on original frames"""
    T = len(frames)
    H, W = edges.shape[1:3]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, (W, H))
    
    for t in tqdm(range(min(T, len(edges))), desc="creating overlay video"):
        frame = frames[t].copy()
        
        #overlay edges in green
        edge_mask = edges[t] > 0
        frame[edge_mask] = [0, 255, 0]
        
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()
    print(f"saved overlay video to {output_path}")


if __name__ == "__main__":
    from video_scaler import scale_video, load_scaled_video
    
    video_path = os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4")
    scaled_path = os.path.join(config.OUTPUT_DIR, "scaled_video.mp4")
    
    if os.path.exists(video_path):
        if not os.path.exists(scaled_path):
            frames, _ = scale_video(video_path, scaled_path)
        else:
            frames = load_scaled_video(scaled_path)
        
        edge_results = process_edges(frames)
        
        feature_path = os.path.join(config.OUTPUT_DIR, "feature_map.mp4")
        create_feature_video(frames, edge_results['filtered'], feature_path)
        
        overlay_path = os.path.join(config.OUTPUT_DIR, "edge_overlay.mp4")
        create_overlay_video(frames, edge_results['filtered'], overlay_path)
    else:
        print(f"video not found: {video_path}")
