#configuration constants for bump detection pipeline with EM training

import os

#paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
MODEL_DIR = "models"
TRAINING_DATA_DIR = "training_data"

#video parameters (step 0)
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
TARGET_FPS = 15
ORIGINAL_FPS = 30  #assumed, will be read from video

#video scaling parameters (step 0)
LOWPASS_SIGMA = 0.8  #mild blur before downscale
SHARPEN_AMOUNT = 0.5  #mild edge sharpening after downscale
SHARPEN_RADIUS = 5  #unsharp mask radius

#audio parameters for bump candidates (step 1)
AUDIO_SAMPLE_RATE = 44100
AUDIO_HIGHPASS_FREQ_HZ = 4000  #highpass cutoff - bumps have energy above this
ENERGY_THRESHOLD_PERCENTILE = 99.5
MIN_BUMP_DISTANCE_SEC = 0.5  #merge candidates closer than this

#candidate window parameters (step 2)
CANDIDATE_RADIUS = 5  #frames before/after audio candidate (r parameter)
GAUSSIAN_SIGMA = 2.0  #sigma for gaussian prior over offsets

#clip extraction parameters (step 3)
EARLY_WARNING_HORIZON = 8  #H - predict bump this many frames ahead
CLIP_LENGTH = 15  #T_clip - frames per clip
MIN_NEGATIVE_DISTANCE = 15  #frames away from any candidate for negatives
POSITIVE_SAMPLES_PER_CANDIDATE = 3  #sample 1 clip per candidate (use prior peak)
CLASS_BALANCE_RATIO = 1.0  #target ratio of negatives to positives

#edge detection parameters (step 4)
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
EDGE_BLUR_SIGMA = 1.0  #optional blur on edges
MORPHO_KERNEL_SIZE = 3  #for thinning/dilation
EDGE_CHANNELS = 3  #thin edges, thick edges, edge magnitude

#model parameters (step 5)
INPUT_CHANNELS = 4  #rgb + edge channel
INPUT_FRAMES = CLIP_LENGTH
BASE_FILTERS = 64
DROPOUT = 0.3
BACKBONE = 'resnet18'  #resnet18, resnet34, efficientnet_b0, mobilenet_v3
TEMPORAL_HEAD = 'gru'  #gru, cnn, lstm
HIDDEN_DIM = 256
FREEZE_EARLY_LAYERS = True  #freeze early pretrained layers

#training parameters
BATCH_SIZE = 24
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
TRAIN_VAL_SPLIT = 0.8
GRAD_CLIP = 1.0

#em training parameters (steps 6-7)
EM_ITERATIONS = 1  #number of E-M cycles (1 = no EM, just prior weights)
EM_EPOCHS_PER_M_STEP = 3  #epochs per M-step
LAMBDA_NEG = 1.0  #weight for negative class in loss
RESPONSIBILITY_TEMPERATURE = 1.0  #temperature for softmax in E-step
EM_WEIGHT_MOMENTUM = 0.5  #blend old/new weights (0=new only, higher=smoother)

#inference parameters (step 8)
DETECTION_THRESHOLD = 0.5  #lowered for better recall
INFERENCE_STRIDE = 1  #sliding window stride
NMS_WINDOW = 10  #frames for non-maximum suppression
EVAL_TOLERANCE_FRAMES = 15  #tolerance for matching detections to ground truth (1 sec at 15fps)

#focal loss parameters
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

#create directories
for d in [OUTPUT_DIR, MODEL_DIR, TRAINING_DATA_DIR]:
    os.makedirs(d, exist_ok=True)
