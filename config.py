from easydict import EasyDict

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

args = EasyDict({
    'n_gpu':1,
    'image_size':384,
    'patch_size':16,
    'device':'cuda',
    'batch_size':16,
    'num_workers':16,
    'epochs':20,
    'lr':2e-4,
    'wd':1e-5,
    'obj':'grid',
    'val_ratio':0.4,
    'save_dir':'./saved_results',
    'dataset_path': '/home/phuocddat/git-repo/cv/ADer/data/mvtec',
    'model':'vit',
    'amp':True,
    'seed':42,
    'beta1':0.5,
    'beta2':0.999
})
