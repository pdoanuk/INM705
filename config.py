from easydict import EasyDict

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

args = EasyDict({
    'n_gpu': 1,
    'gpus': 'cuda',
    'image_size': 256,
    'patch_size': 16,
    'device': 'cuda:0',
    'batch_size': 16,
    'num_workers': 16,
    'epochs': 100,
    'val_epochs': 20,
    'lr': 2e-4,
    'weight_decay': 0.0001,
    'obj': 'all',
    'val_ratio': 0.2,
    'save_dir': './saved_results/',
    'dataset_path': '/home/phuocddat/git-repo/cv/ADer/data/mvtec',
    'model': 'vit',
    'amp': True,
    'seed': 42,
    'beta1': 0.9,
    'beta2': 0.999,
    'early_stopping_patience': 12,
    'wandb_log': True,
    'mode': "online",
    'project': "INM705 - EXP",
    'notes': 'AdamW, L2Loss',  # Notes about the run, verbose description [str]
    'log_comment': ' Some comment',  # Comment to add to name the local logging folder [str]
    'wandb_dir': './wandb_log',  # Direcotry to store the wandb file. CAREFUL: Directory must exists [str]
    'log_dir': './logdir',  # Direcotry to store all logging related files and outputs [str]
    'level': 'info',  # Level of logging must be either ["critical", "error", "warning", "info", "debug"] [str]
    'log_images': True,  # If images should be logged to WandB for this run. [bool] [Optional, defaults to False]

})

