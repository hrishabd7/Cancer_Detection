import os
import random
import numpy as np
import torch

# ========= Paths =========
ROOT      = os.path.join(os.getcwd(), "data", "Data")
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR   = os.path.join(ROOT, "valid")
TEST_DIR  = os.path.join(ROOT, "test")

CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_path = os.path.join(CHECKPOINT_DIR, "best_convnext_tiny.pt")
last_path = os.path.join(CHECKPOINT_DIR, "last_convnext_tiny.pt")
swa_path  = os.path.join(CHECKPOINT_DIR, "swa_convnext_tiny.pt")

# ========= Hyperparameters =========
IMG_SIZE   = 512
BATCH      = 15
EPOCHS     = 15
EPOCHS_FT  = 25
LR_HEAD    = 5e-4
LR_FT      = 5e-5
WD         = 1e-4
SMOOTH_START = 0.02
SMOOTH_END   = 0.0
SWA_EPOCHS   = 5
UNCERTAIN_THRESH = 0.60
USE_FOCAL   = False
SEED = 42

# ========= Reproducibility =========
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ========= Paths & Config =========
# ROOT      = os.path.join(os.getcwd(), "data", "Data")
# TRAIN_DIR = os.path.join(ROOT, "train")
# VAL_DIR   = os.path.join(ROOT, "valid")
# TEST_DIR  = os.path.join(ROOT, "test")

# IMG_SIZE   = 512         # higher resolution for subtle lung patterns
# BATCH      = 10          # adjust down if OOM on MPS/CPU
# EPOCHS     = 1          # phase 1: head warmup
# EPOCHS_FT  = 1          # phase 2: full fine-tune
# LR_HEAD    = 5e-4
# LR_FT      = 5e-5
# WD         = 1e-4        # AdamW weight decay
# SMOOTH_START = 0.02      # label smoothing at start
# SMOOTH_END   = 0.0       # anneal to 0 near the end
# SWA_EPOCHS   = 5         # last N FT epochs use SWA averaging
# UNCERTAIN_THRESH = 0.60  # flag low-confidence predictions
# USE_FOCAL   = False      # set True if cancer-vs-cancer confusion persists
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# best_path = os.path.join(CHECKPOINT_DIR, "best_convnext_tiny.pt")
# last_path = os.path.join(CHECKPOINT_DIR, "last_convnext_tiny.pt")
# swa_path  = os.path.join(CHECKPOINT_DIR, "swa_convnext_tiny.pt")