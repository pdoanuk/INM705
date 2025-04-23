from easydict import EasyDict

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

ANOMALY_MAP_SIGMA = 4 # Need to revise the impact
# Metrics to compute using the Evaluator
METRICS_TO_COMPUTE = [
    'mAUROC_px', 'mAUROC_sp_max', # Pixel and Image AUROC
    'mAUPRO_px',                 # Pixel AUPRO
    'mAP_px', 'mAP_sp_max',      # Pixel and Image Average Precision
    'mF1_max_sp_max',            # Max F1 for Image (max pooling)
    'mIoU_max_px',               # Max Pixel IoU over thresholds
]
WANDB_PROJECT_NAME = "INM705-exp" # Consider making this configurable via args
WANDB_TAGS = ["experiment", "ViT-AD", "ViTAD custom model"]
# Define identifier for the full dataset run in args.obj
FULL_DATASET_IDENTIFIER = "full" # Need to review this later, configuration may be: single object, full[sep, unified]
DEFAULT_IMAGE_TO_VISUAL = [1, 5, 10, 15]

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

args = EasyDict({
    'n_gpu': 1,
    'gpus': 'cuda',
    'use_adeval': True,
    'image_size': 256,
    'patch_size': 16,
    'device': 'cuda:0',
    'batch_size': 16,
    'num_workers': 16,
    'epochs': 100,
    'warmup_epochs': 2,
    'val_epochs': 10,
    'loss_func': 'CosLoss', # CosLoss, L2Loss
    'lr': 1e-3 * 16 / 8, #0.001, # was 2e-4
    'weight_decay': 0.0001,
    'obj': 'full', # all: run all classes separate, full run all in one
    'val_ratio': 0.2,
    'save_dir': './saved_results_new/',
    'dataset_path': '/home/phuocddat/git-repo/cv/ADer/data/mvtec',
    'model': 'ViTAD_Fusion_v2', # 'ViTAD_Fusion', VitDecoderExp, 'ViTAD_Fusion_v2'
    'amp': True,
    'seed': 42,
    'beta1': 0.9,
    'beta2': 0.999,
    'early_stopping_patience': 12,
    'wandb_log': True,
    'mode': "online",
    'project': "INM705 - EXP",
    'notes': 'AdamW, LRScheduler Cosine - adeEval enable. Evaluate every val_epoch _ Separate classes VitDecoderExp',  # Notes about the run, verbose description [str]
    'log_comment': 'Test mnew metrics--- VitDecoderExp Full combined dataset, CosLoss,',  # Comment to add to name the local logging folder [str]
    'wandb_dir': './wandb_log',  # Direcotry to store the wandb file. CAREFUL: Directory must exists [str]
    'log_dir': './logdir',  # Direcotry to store all logging related files and outputs [str]
    'level': 'info',  # Level of logging must be either ["critical", "error", "warning", "info", "debug"] [str]
    'log_images': True,  # If images should be logged to WandB for this run. [bool] [Optional, defaults to False]

})

