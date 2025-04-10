import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import numpy as np
from dataset import MVTecDataset
import torch.optim as optim
from model_vit import ViTAutoencoder
import torch
import torch.nn as nn
import timm
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

## TIMESTAMP @ 2025-04-10T23:45:47
## author: phuocddat
## start
# very basic pipeline to work
## end --


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 50  # Adjust as needed
root = '../data/mvtec/'
category = 'bottle'
train_dataset = MVTecDataset(root, category, is_train=True)
test_dataset = MVTecDataset(root, category, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, _) in enumerate(train_loader):  # Only need images for training
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # Ensure inputs and outputs are compatible for loss (e.g., both normalized)
        loss = criterion(outputs, inputs)  # MSE between normalized input and reconstruction
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

# Save the trained model weights
torch.save(model.state_dict(), f'{category}_vit_ae_baseline.pth')
print(f"Save complete.")


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


# Image-level AUROC
