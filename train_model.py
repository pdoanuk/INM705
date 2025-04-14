## TIMESTAMP @ 2025-04-10T23:45:47
## author: phuocddat
## start
# very basic pipeline to work
## end --

import datetime
import os
import copy
import numpy as np
import timm
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim

from easydict import EasyDict
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset_mvtec import get_dataloader
from model_vit import ViTAutoencoder
from utils_mvtec import *
from config import args, CLASS_NAMES, mean_train, std_train
from matplotlib import pyplot as plt

# Set up default parameters


log_run_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir= f"./saved_results/{log_run_time}"
os.makedirs(save_dir, exist_ok=True)
# if not os.path.exists(args.save_dir):
#     os.makedirs(save_dir, exist_ok=True)
print(f"Saving to {save_dir}")
log_path = os.path.join(save_dir, 'log_{}_{}.txt'.format(args.obj, args.model))
log = open(log_path, 'w')
print(f"Logging to {log_path}")

random_seed = 42
set_seed(random_seed)

# Setup default working environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = amp.GradScaler()

# Initiate model
model = ViTAutoencoder().to(device)
# Optimizer
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = optim.Adam(params=model.parameters(),
                       lr=args.lr,
                       betas=(args.beta1, args.beta2)
                       )
criterion = nn.MSELoss()

# Load data loaders

train_data, val_data, test_data = get_dataloader(args)

# Define train pipeline
def train_pipeline(args, scaler, model, epoch, train_loader, optimizer, log):
    model.train()
    MSE = nn.MSELoss()

    for (x, _, _) in tqdm(train_loader):
        x = x.to(args.device)
        optimizer.zero_grad()
        if args.amp:
            with amp.autocast():
                x_hat = model(x)
                loss = MSE(x, x_hat)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    log_write('Train Epoch: {} | MSE Loss: {:.6f}'.format(epoch, loss), log)


# Define validation pipeline
def val_pipeline(args, model, epoch, val_loader, log):
    model.eval()
    MSE = nn.MSELoss()

    for (x, _, _) in tqdm(val_loader):
        x = x.to(args.device)
        with torch.no_grad():
            x_hat = model(x)
            loss = MSE(x, x_hat)

    log_write(('Valid Epoch: {} | MSE Loss: {:.6f}'.format(epoch, loss)), log)

    return loss, model

# Define export visualisation for debug procedure
def save_debug_image(test_loader, test_imgs, recon_imgs, mean, std, seg_scores, gt_mask_list):
    for num in range(len(test_loader)):
        if num in [5, 10, 15]:
            if test_imgs[num].dtype != "uint8":
                test_imgs[num] = denormalization(test_imgs[num], mean, std)

            if recon_imgs[num].dtype != "uint8":
                recon_imgs[num] = denormalization(recon_imgs[num], mean, std)

            scores_img = seg_scores[num]
            fig, plots = plt.subplots(1, 4)
            fig.set_figwidth(9)
            fig.set_tight_layout(True)
            plots = plots.reshape(-1)
            plots[0].imshow(test_imgs[num])
            plots[1].imshow(recon_imgs[num])
            plots[2].imshow(scores_img, cmap='jet', alpha=0.35)
            plots[3].imshow(gt_mask_list[num], cmap=plt.cm.gray)

            plots[0].set_title("Real image:")
            plots[1].set_title("Reconstructed image")
            plots[2].set_title("Anomaly map")
            plots[3].set_title("GT mask")
            plt.savefig(f"{save_dir}/test_image_{args.model}_{args.obj}_{num}.png")


def full_test_pipeline(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    MSE = nn.MSELoss(reduction='none')
    det_scores, seg_scores = [],[]
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []

    det_sig, seg_sig = 15,6
    for (x, label, mask) in tqdm(test_loader):
        mask = mask.squeeze(0)
        test_imgs.extend(x.cpu().numpy())
        gt_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())
        score = 0
        with torch.no_grad():
            x = x.to(device)
            x_hat = model(x)
            mse = MSE(x,x_hat)
            score = mse

        score = score.cpu().numpy()
        score = score.mean(1) #

        det_score, seg_score = copy.deepcopy(score), copy.deepcopy(score)

        for i in range(det_score.shape[0]):
            det_score[i] = gaussian_filter(det_score[i], sigma=det_sig)
        det_scores.extend(det_score)

        for i in range(seg_score.shape[0]):
            seg_score[i] = gaussian_filter(seg_score[i], sigma=seg_sig)
        seg_scores.extend(seg_score)

        recon_imgs.extend(x_hat.cpu().numpy())
    return det_scores, seg_scores, test_imgs, recon_imgs, gt_list, gt_mask_list


# Start training procedure

for epoch in range(1, args.epochs + 1):
    log_write('Epoch: {:3d}/{:3d} '.format(epoch, args.epochs), log)
    train_pipeline(args=args,
                   scaler=scaler,
                   model=model,
                   epoch=epoch,
                   train_loader=train_data,
                   optimizer=optimizer,
                   log=log)

    if epoch % 10 == 0:
        val_loss, save_model = val_pipeline(args=args, model=model, epoch=epoch, val_loader=val_data, log=log)


log.close()
final_model_name = os.path.join(save_dir, 'model_{}_{}_final_epoch_model.pt'.format(args.obj, args.model))
torch.save(save_model.state_dict(), final_model_name)
# Release model
model = None
## Reload model for evaluating and testing
model_eval = ViTAutoencoder()
model_eval.load_state_dict(torch.load(final_model_name))
model_eval.to(device)

# Get test in processing
det_scores, seg_scores, test_imgs, recon_imgs, gt_list, gt_mask_list = full_test_pipeline(model= model_eval,
                                                                                          test_loader=test_data)

seg_scores = np.asarray(seg_scores)
max_anomaly_score = seg_scores.max()
min_anomaly_score = seg_scores.min()
seg_scores = (seg_scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

gt_mask = np.asarray(gt_mask_list)
gt_mask = gt_mask.astype('int')
per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), seg_scores.flatten())
print('pixel ROCAUC: %.2f' % (per_pixel_rocauc))

