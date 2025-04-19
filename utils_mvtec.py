import random
from tqdm import tqdm
import random

import numpy as np
import torch
from easydict import EasyDict
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



def calculate_pro_auc(anomaly_maps, ground_truth_masks):
    aucs = []
    # Normalize anomaly maps per image
    max_vals = np.max(anomaly_maps.reshape(anomaly_maps.shape[0], -1), axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
    min_vals = np.min(anomaly_maps.reshape(anomaly_maps.shape[0], -1), axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
    norm_anomaly_maps = (anomaly_maps - min_vals) / (max_vals - min_vals + 1e-8)

    for i in range(len(ground_truth_masks)):
        gt = ground_truth_masks[i].flatten()
        pred = norm_anomaly_maps[i].flatten()
        # Only calculate if there is an anomaly in ground truth
        if np.sum(gt) > 0:
            try:
                aucs.append(roc_auc_score(gt, pred))
            except ValueError: # Handle cases with only one class in gt (shouldn't happen if sum(gt)>0)
                 aucs.append(0.0) # Or handle appropriately

    return np.mean(aucs) if aucs else 0.0 #


def denormalization(x, mean, std):
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)
    return x


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def log_write(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()