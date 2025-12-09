#step 4: edge detection and multi-channel feature stacking
#creates edge features: thin edges, thick edges, edge magnitude

import cv2
import numpy as np
from tqdm import tqdm
import os

import config


def frame_to_grayscale(frame):
    """convert rgb frame to grayscale"""
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def frame_to_luma(frame):
    """convert rgb frame to luma (Y from YUV, better for edge detection)"""
    if len(frame.shape) == 3:
        #bt.709 luma coefficients
        return (0.2126 * frame[:,:,0] + 0.7152 * frame[:,:,1] + 
                0.0722 * frame[:,:,2]).astype(np.uint8)
    return frame


def compute_edge_magnitude(gray):
    """compute edge magnitude using sobel operators"""
    #apply gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), config.EDGE_BLUR_SIGMA)
    
    #sobel gradients
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    #magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    #normalize to 0-255
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    
    return magnitude


def canny_edge_detection(frame, low_thresh=None, high_thresh=None):
    """apply canny edge detection to a single frame"""
    low_thresh = low_thresh or config.CANNY_LOW_THRESHOLD
    high_thresh = high_thresh or config.CANNY_HIGH_THRESHOLD
    
    gray = frame_to_luma(frame)
    blurred = cv2.GaussianBlur(gray, (5, 5), config.EDGE_BLUR_SIGMA)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    return edges


def thin_edges(edges, kernel_size=None):
    """morphological thinning of edges"""
    kernel_size = kernel_size or config.MORPHO_KERNEL_SIZE
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    #erosion to thin
    thinned = cv2.erode(edges, kernel, iterations=1)
    
    return thinned


def thick_edges(edges, kernel_size=None):
    """morphological dilation of edges"""
    kernel_size = kernel_size or config.MORPHO_KERNEL_SIZE
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    #dilation to thicken
    thickened = cv2.dilate(edges, kernel, iterations=1)
    
    return thickened


def process_frame_edges(frame):
    """
    process single frame to extract multi-channel edge features
    
    returns (H, W, 3) array with:
        channel 0: thin edges (morphological erosion of canny)
        channel 1: thick edges (morphological dilation of canny)  
        channel 2: edge magnitude (sobel gradient)
    """
    gray = frame_to_luma(frame)
    
    #canny edges
    canny = canny_edge_detection(frame)
    
    #channel 0: thin edges
    thin = thin_edges(canny)
    
    #channel 1: thick edges
    thick = thick_edges(canny)
    
    #channel 2: edge magnitude
    magnitude = compute_edge_magnitude(gray)
    
    #stack channels
    features = np.stack([thin, thick, magnitude], axis=-1)
    
    return features


def process_edge_features(frames, show_progress=True):
    """
    step 4: process all frames to extract edge features
    
    args:
        frames: video frames (T, H, W, C)
        show_progress: show progress bar
    
    returns:
        features: (T, H, W, 3) array with edge channels
    """
    print("step 4: extracting multi-channel edge features...")
    print(f"  thin edges (canny + erosion)")
    print(f"  thick edges (canny + dilation)")
    print(f"  edge magnitude (sobel gradient)")
    
    features = []
    iterator = tqdm(frames, desc="edge features") if show_progress else frames
    
    for frame in iterator:
        feat = process_frame_edges(frame)
        features.append(feat)
    
    features = np.array(features)
    print(f"  extracted features: {features.shape}")
    
    return features


def create_input_tensor(frames, features):
    """
    create model input by stacking luma and edge features
    
    args:
        frames: rgb frames (T, H, W, 3)
        features: edge features (T, H, W, 3)
    
    returns:
        tensor: (T, H, W, 4) with [luma, thin, thick, magnitude]
    """
    #extract luma from frames
    luma = np.array([frame_to_luma(f) for f in frames])
    luma = luma[..., np.newaxis]  #(T, H, W, 1)
    
    #combine luma with edge features
    tensor = np.concatenate([luma, features], axis=-1)
    
    return tensor


def create_rgb_edge_tensor(frames, features):
    """
    create model input by stacking rgb and single edge channel
    
    args:
        frames: rgb frames (T, H, W, 3)
        features: edge features (T, H, W, 3)
    
    returns:
        tensor: (T, H, W, 4) with [R, G, B, canny_edges]
    """
    #use thick edges as the primary edge channel
    edges = features[..., 1:2]  #thick edges
    
    combined = np.concatenate([frames, edges], axis=-1)
    
    return combined


def process_edges(frames):
    """legacy function for backwards compatibility"""
    features = process_edge_features(frames)
    return {
        'raw_edges': features[..., 0],  #thin edges
        'filtered': features[..., 1],   #thick edges
        'features': features
    }


def create_feature_video(frames, features, output_path):
    """create video showing the edge features as RGB"""
    T, H, W, C = features.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, (W, H))
    
    for t in tqdm(range(T), desc="creating feature video"):
        #use features as RGB channels
        frame = features[t]  #(H, W, 3)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()
    print(f"saved feature video to {output_path}")


def create_overlay_video(frames, features, output_path):
    """create video with edge overlay on original frames"""
    T = len(frames)
    H, W = features.shape[1:3]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, (W, H))
    
    for t in tqdm(range(min(T, len(features))), desc="creating overlay video"):
        frame = frames[t].copy()
        
        #overlay thick edges in green
        edge_mask = features[t, :, :, 1] > 0  #thick edges channel
        frame[edge_mask] = [0, 255, 0]
        
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()
    print(f"saved overlay video to {output_path}")


def visualize_edge_channels(frame, features, save_path=None):
    """visualize all edge channels for a single frame"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(frame)
    axes[0, 0].set_title('original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(features[:, :, 0], cmap='gray')
    axes[0, 1].set_title('thin edges')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(features[:, :, 1], cmap='gray')
    axes[1, 0].set_title('thick edges')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(features[:, :, 2], cmap='gray')
    axes[1, 1].set_title('edge magnitude')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved edge visualization to {save_path}")
    
    plt.close(fig)
    return fig


if __name__ == "__main__":
    from video_scaler import scale_video, load_scaled_video
    
    video_path = os.path.join(config.DATA_DIR, "PXL_20251118_131050616.TS.mp4")
    scaled_path = os.path.join(config.OUTPUT_DIR, "scaled_video.mp4")
    
    if os.path.exists(video_path):
        if not os.path.exists(scaled_path):
            scale_video(video_path, scaled_path)
        
        frames = load_scaled_video(scaled_path, max_frames=1000)
        
        features = process_edge_features(frames)
        
        #visualize single frame
        viz_path = os.path.join(config.OUTPUT_DIR, "edge_channels.png")
        visualize_edge_channels(frames[500], features[500], save_path=viz_path)
        
        #create videos
        feature_path = os.path.join(config.OUTPUT_DIR, "feature_map.mp4")
        create_feature_video(frames, features, feature_path)
        
        overlay_path = os.path.join(config.OUTPUT_DIR, "edge_overlay.mp4")
        create_overlay_video(frames, features, overlay_path)
    else:
        print(f"video not found: {video_path}")
