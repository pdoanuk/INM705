# -*- coding: utf-8 -*-
"""
Configuration constants and default hyperparameter values for MVTec AD Pipeline.
Defaults can be overridden by command-line arguments or wandb sweeps.
"""
import argparse

# --- Non-Tunable Constants ---
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

METRICS_TO_COMPUTE = [
    'mAUROC_px', 'mAUROC_sp_max', # Pixel and Image AUROC
    'mAUPRO_px',                 # Pixel AUPRO
    'mAP_px', 'mAP_sp_max',      # Pixel and Image Average Precision
    'mF1_max_sp_max',            # Max F1 for Image (max pooling)
    'mIoU_max_px',               # Max Pixel IoU over thresholds
]
PRIMARY_KEY_METRIC = 'average/mAUROC_px'
ANOMALY_MAP_SIGMA = 4
DEFAULT_IMAGE_TO_VISUAL = [1, 5, 10, 15]

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

FULL_DATASET_IDENTIFIER = 'full'
SINGLE_CLASS_MODE = "title" #
ALL_CLASSES_SEPARATELY_MODE = "full" # all. full SINGLE_CLASS_NAME - SWEEP

# --- Default Configuration Values (will be parsed by argparse) ---
DEFAULT_CONFIG = {
    # --- Hardware ---
    'n_gpu': 1,
    'gpus': 'cuda', # Not directly used if device is set
    'device': 'cuda:0',
    'num_workers': 16,
    'amp': True, # Mixed precision

    # --- Data & Model ---
    'dataset_path': '/home/phuocddat/git-repo/cv/ADer/data/mvtec',
    'obj': ALL_CLASSES_SEPARATELY_MODE, # Default: run all classes separately
    'image_size': 256,
    'patch_size': 16, # Tunable - ViT parameter
    'model': 'ViTAD_Fusion_v2', # Tunable - Model variant ('VitDecoderExp', 'ViTAD_Fusion', 'ViTAD_Fusion_v2) Sweep
    'val_ratio': 0.2, # Validation split ratio (if applicable in dataset loading)

    # --- Training ---
    'epochs': 100, # Tunable
    'batch_size': 16, # Tunable 8, 16, 32
    'warmup_epochs': 2,
    'val_epochs': 10,
    'early_stopping_patience': 12,

    # --- Optimizer & Loss ---
    'optimizer': 'AdamW', # Tunable ('AdamW', 'Adam')
    'lr': 1e-4 * 16 / 8, # Maybe simplify default to 2e-4
    'weight_decay': 1e-4, # Tunable
    'beta1': 0.9, # Tunable (Adam/AdamW param)
    'beta2': 0.999, # Tunable (Adam/AdamW param)
    'loss_func': 'L2Loss', # Tunable ('L2Loss', 'CosLoss')

    # --- Evaluation & Debugging ---
    'use_adeval': True, # Use external AD-Eval library if available
    'anomaly_map_sigma': 4, # Sigma for Gaussian smoothing of anomaly map
    'metrics_to_compute': METRICS_TO_COMPUTE,
    'debug_image_indices': DEFAULT_IMAGE_TO_VISUAL, # Indices for debug images

    # --- Logging & Saving ---
    'seed': 42,
    'save_dir': './saved_results_sweep/', # Base directory for saving results
    'wandb_log': True,
    'mode': "online",
    'project': "INM705 - EXP - SWEEP",
    'tags': ["sweep", "ViT-AD", "ViTAD"],
    'notes': 'AdamW, LRScheduler Cosine - adeEval enable. Evaluate every val_epoch _ Separate classes VitDecoderExp',  # Notes about the run, verbose description [str]
    'log_comment': 'Test mnew metrics--- VitDecoderExp Full combined dataset, CosLoss,',  # Comment to add to name the local logging folder [str]
    'wandb_dir': './wandb_log',  # Direcotry to store the wandb file. CAREFUL: Directory must exists [str]
    'log_dir': './logdir',  # Direcotry to store all logging related files and outputs [str]
    'level': 'info',  # Level of logging must be either ["critical", "error", "warning", "info", "debug"] [str]
    'log_images': True,  # If images should be logged to WandB for this run. [bool] [Optional, defaults to False]
}

# --- Helper to get args object ---
parser = argparse.ArgumentParser()
for key, value in DEFAULT_CONFIG.items():
    parser.add_argument(f'--{key}', type=type(value), default=value)
args = parser.parse_args()