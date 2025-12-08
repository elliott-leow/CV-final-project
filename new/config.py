#configuration constants for bump detection pipeline

import os

#paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
MODEL_DIR = "models"
TRAINING_DATA_DIR = "training_data"

#video parameters
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
TARGET_FPS = 15
ORIGINAL_FPS = 30  #assumed, will be read from video

#audio parameters for ground truth
AUDIO_SAMPLE_RATE = 44100
MIN_FREQ_HZ = 100  #low frequency threshold for bump detection
MAX_FREQ_HZ = 2000  #high frequency threshold
ENERGY_THRESHOLD_PERCENTILE = 85  #percentile for bump energy threshold
MIN_BUMP_DISTANCE_SEC = 0.3  #minimum time between bumps

#training clip parameters
FRAMES_BEFORE_BUMP = 8  #frames to look back before bump (detect 8 frames before bump)
CLIP_LENGTH = 15  #total frames per training clip
MIN_FRAMES_TO_NEXT_BUMP = 30  #minimum frames to next bump for negative samples

#edge detection parameters
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
TEMPORAL_KERNEL_SIZE = 3  #frames for morphological operations
MIN_EDGE_TEMPORAL_LENGTH = 3  #minimum frames for edge persistence
MAX_EDGE_ANGLE_FROM_HORIZONTAL = 70  #degrees

#model parameters
INPUT_CHANNELS = 3
INPUT_FRAMES = 15
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
TRAIN_VAL_SPLIT = 0.8

#detection parameters
DETECTION_THRESHOLD = 0.7  #confidence threshold for detection

#spikey loss parameters (for better peak detection)
FOCAL_ALPHA = 0.25     #positive class weight in focal loss
FOCAL_GAMMA = 2.0      #focus on hard examples (higher = more focus)
MARGIN_WEIGHT = 0.3    #weight for confidence margin loss
POS_TARGET = 0.9       #target probability for bump frames
NEG_TARGET = 0.1       #target probability for non-bump frames

#create directories
for d in [OUTPUT_DIR, MODEL_DIR, TRAINING_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

