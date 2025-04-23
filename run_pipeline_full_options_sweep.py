# -*- coding: utf-8 -*-
"""
MVTec AD Anomaly Detection Training and Evaluation Pipeline
Using Vision Transformer (ViT) Encoder-Decoder Variants (ViTAD).

Supports training on single MVTec classes or the full combined dataset.

TIMESTAMP @ 2025-04-10T23:45:47 (Revised: 2025-04-19)
author: phuocddat (Refactored with metric.py integration and full dataset support)
"""

import datetime
import os
import copy
import time
import sys
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import wandb
import logging # Use standard logging
from scipy.ndimage import gaussian_filter
from pathlib import Path
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from typing import Tuple, List, Dict, Any, Optional, Union
from PIL import Image


# --- Local Imports ---
try:
    from dataset_mvtec import get_dataloader, get_loader_full
    from model_vitad import load_default_model # Using ViTAD model with manual loading function
    from model_vit import ViTAutoencoder, VitDecoderExp
    from losses import L2Loss, CosLoss, KLLoss # L1Loss, CosLoss also available
    from utils_mvtec import set_seed, denormalization, log_write # Keep utility functions
    from config_sweep import *
    from metrics import Evaluator
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# --- Configuration ---
# SHOULD MOVE TO CONFIG.PY
FULL_DATASET_IDENTIFIER = 'full'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEBUG = True

def get_model(args: argparse.Namespace ,device: torch.device) -> nn.Module:
    """Instantiates and returns the anomaly detection model.

    Args:
        args: args
        device:
    """
    """This one is only for ViTAD-based model"""
    if args.model == "ViTAD_Fusion":
        model = load_default_model().to(device)
    elif args.model == "VitDecoderExp":
        model = VitDecoderExp().to(device)
    else:
        logger.info(f"Using model {args.model} was not supported yet")
        sys.exit(1)
    if model is not None:
        logger.info(f"Using model: {model.__class__.__name__} loading on {device}")
    if DEBUG:
        try:
            summary(model, input_size=(args.batch_size, 3, args.image_size, args.image_size), device=device)
        except Exception as e:
            logger.warning(f"Could not generate model summary: {e}")
    return model

def get_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    """Instantiates and returns the optimizer based on args."""
    lr = args.lr
    wd = args.weight_decay
    beta1 = args.beta1
    beta2 = args.beta2

    if args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd, amsgrad=False)
        logger.info(f"Using optimizer: AdamW with LR: {lr}, WD: {wd}")
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)
        logger.info(f"Using optimizer: Adam with LR: {lr}, WD: {wd}")
    else:
        logger.error(f"Unsupported optimizer type: {args.optimizer}")
        sys.exit(1)
    return optimizer

# def get_optimizer(model: nn.Module) -> optim.Optimizer:
#     """Instantiates and returns the optimizer.
#
#     Args:
#         model:
#     """
#     # todo: MOVE TO DEDICATED FUNCTION LATER
#     optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay,
#                             amsgrad=False)
#     logger.info(f"Using optimizer: {optimizer.__class__.__name__} with LR: {args.lr}, Weight Decay: {args.weight_decay}")
#     return optimizer

def get_criterion(args: argparse.Namespace) -> nn.Module:
    """Instantiates and returns the loss function based on args."""
    if args.loss_func.lower() == 'l2loss':
        criterion = L2Loss()
        logger.info("Using loss function: L2Loss")
    elif args.loss_func.lower() == 'cosloss':
        criterion = CosLoss()
        logger.info("Using loss function: CosLoss")
    else:
        logger.error(f"Unsupported loss function: {args.loss_func}")
        sys.exit(1)
    return criterion

def train_epoch(
    args: argparse.Namespace,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: amp.GradScaler,
    device: torch.device,
    epoch: int,
    run_context: str = "" # e.g., class name or "full"
) -> float:
    """

    Args:
        args:
        model:
        dataloader:
        optimizer:
        criterion:
        scaler:
        device:
        epoch:
        run_context:

    Returns:

    """
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [{run_context} Train]", leave=False)

    # Correctly unpack data from dataloader
    for x, y, mask, cls_name in pbar:
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True) # More efficient zeroing

        if args.amp:
            with amp.autocast():
                if args.model == "ViTAD_Fusion":
                    feature_enc, feature_fus = model(x)  # Get features from ViTAD model
                    # Only use features relevant for the loss (ViTAD compares enc/fus)
                    loss = criterion(feature_enc, feature_fus)  # Compare features
                else:
                    x_hat = model(x)
                    #loss = criterion(x_hat, x)
                    loss = criterion(x, x_hat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.model == "ViTAD_Fusion":
                feature_enc, feature_fus = model(x)
                loss = criterion(feature_enc, feature_fus)
            else:
                x_hat = model(x)
                #loss = criterion(x_hat, x)
                loss = criterion(x, x_hat)

            loss.backward()
            optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        pbar.set_postfix(loss=f"{batch_loss:.6f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_epoch(
    args: argparse.Namespace,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    run_context: str = "" # e.g., class name or "full"
) -> float:
    """
    Validate process
    Args:
        args: args from config.py
        model: model instance
        dataloader: dataloader should be prepared in advanced.
        criterion: loss function
        device: device instance (gpu/cpu)
        epoch: to be removed later
        run_context: to be removed later

    Returns:

    """
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [{run_context} Val]", leave=False)

    with torch.no_grad():
        # Correctly unpack data from dataloader
        for x, y, mask, cls_name in pbar:
            x = x.to(device)
            if args.amp:
                with amp.autocast():
                    if args.model == "ViTAD_Fusion":
                        feature_enc, feature_fus = model(x)  # Get features from ViTAD model
                        # Only use features relevant for the loss (ViTAD compares enc/fus)
                        loss = criterion(feature_enc, feature_fus)  # Compare features
                    else:
                        x_hat = model(x)
                        #loss = criterion(x_hat, x)
                        loss = criterion(x, x_hat)
                        # feature_enc, feature_fus = model(x)
                    # loss = criterion(feature_enc, feature_fus)
            else:
                if args.model == "ViTAD_Fusion":
                    feature_enc, feature_fus = model(x)  # Get features from ViTAD model
                    # Only use features relevant for the loss (ViTAD compares enc/fus)
                    loss = criterion(feature_enc, feature_fus)  # Compare features
                else:
                    x_hat = model(x)
                    #loss = criterion(x_hat, x)
                    loss = criterion(x, x_hat)

                    # feature_enc, feature_fus = model(x)
                # loss = criterion(feature_enc, feature_fus)

            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(loss=f"{batch_loss:.6f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate_performance(
    args: argparse.Namespace,
    model: nn.Module,
    test_loader: DataLoader,
    epoch_number: Union[int, str, None], # Can be int epoch or 'final'
    device: torch.device,
    save_dir: Path,
    run_context: str, # Class name or 'full' for context
    evaluator: Evaluator, # Pass the evaluator instance
    debug_image_indices: Optional[List[int]] # Indices to save debug images
) -> Dict[str, float]:
    """
    Runs the full test evaluation pipeline using the provided Evaluator.
    Handles both single-class and full-dataset evaluation.

    Args:
        model: The model to evaluate.
        test_loader: DataLoader for the test set.
        epoch_number: Identifier for the evaluation stage (e.g., epoch number or 'final').
        device: The device to run evaluation on.
        save_dir: Directory to save debug images.
        run_context: Context identifier (e.g., specific class name or 'full').
        evaluator: Initialized Evaluator instance.
        debug_image_indices: Indices of images in the test set to save for debugging.

    Returns:
        Dict[str, float]: Dictionary of calculated metrics.
    """
    if DEBUG and debug_image_indices is None:
        debug_image_indices = DEFAULT_IMAGE_TO_VISUAL
        logger.info(f"Set default indices of images for test visualisation: {debug_image_indices}")

    # Make sure mode is set to eval.
    model.eval()

    # --- Data Collection for Evaluator ---
    all_gt_masks = []       # List to store ground truth pixel masks (np arrays)
    all_anomaly_maps = []   # List to store predicted anomaly maps (np arrays)
    all_gt_labels = []      # List to store ground truth image labels (0 or 1)
    all_class_names = []    # List to store actual class names from data
    test_imgs_for_debug = [] # Store original images only if needed for debugging

    epoch_str = str(epoch_number) if epoch_number is not None else 'final'
    logger.info(f"Running evaluation for context: {run_context} (Epoch: {epoch_str})...")
    pbar = tqdm(test_loader, desc=f"Evaluation [{run_context}, Epoch {epoch_str}]", leave=False)
    loss_func = nn.MSELoss(reduction='none') # default for image-wise compute
    output_size = (args.image_size, args.image_size)

    # Correctly unpack data, including the actual class names per batch
    for (x, label, mask, batch_class_names) in pbar:
        current_batch_size = x.size(0)
        if debug_image_indices:
             # Store images for potential debug saving later
             test_imgs_for_debug.extend(list(x.cpu().numpy())) # Store as list of numpy arrays

        x = x.to(device)
        with torch.no_grad():
            if args.amp:
                 with amp.autocast():
                     if args.model == "ViTAD_Fusion":
                         feature_enc, feature_fus = model(x)  # Get features from ViTAD model
                         # Only use features relevant for the loss (ViTAD compares enc/fus)
                     else:
                         x_hat = model(x)
                         #loss_px = loss_func(x_hat, x)
                         loss_px = loss_func(x, x_hat)

                         # feature_enc, feature_fus = model(x) # Get features
            else:
                if args.model == "ViTAD_Fusion":
                    feature_enc, feature_fus = model(x)  # Get features from ViTAD model
                    # Only use features relevant for the loss (ViTAD compares enc/fus)
                else:
                    x_hat = model(x)
                    #loss_px = loss_func(x_hat, x)
                    loss_px = loss_func(x, x_hat)

                # feature_enc, feature_fus = model(x) # Get features

        # --- Anomaly Map Calculation ---
        if args.model == "ViTAD_Fusion":
            if not isinstance(feature_enc, list): feature_enc = [feature_enc]
            if not isinstance(feature_fus, list): feature_fus = [feature_fus]
            # Ensure features are valid before calculation
            if DEBUG:
                if not feature_enc or feature_enc[0] is None or not feature_fus or feature_fus[0] is None:
                    logger.warning(f"Skipping batch due to missing features in evaluation for {run_context}.")
                    continue  # Skip this batch

        try:
            if args.model == "ViTAD_Fusion":
                anomaly_map_batch, _ = Evaluator.cal_anomaly_map(
                    ft_list=feature_enc,
                    fs_list=feature_fus,
                    out_size=output_size,
                    uni_am=False, # Or True, depending on desired fusion strategy
                    amap_mode='add', # Or 'mul'
                    gaussian_sigma=ANOMALY_MAP_SIGMA, # Use configured sigma
                    use_cos=True if args.loss_func == "CosLoss" else False
                )
            else:
                score_map = loss_px.cpu().numpy()
                score_map = score_map.mean(1)
                anomaly_map_batch = copy.deepcopy(score_map)

                for i in range(anomaly_map_batch.shape[0]):
                    anomaly_map_batch[i] = gaussian_filter(anomaly_map_batch[i], sigma=6)
                # anomaly_map_batch.extend(anomaly_map_batch)

                # recon_imgs.extend(x_hat.cpu().numpy())
                #anomaly_map_batch = loss_px.mean(dim=1).cpu().numpy()
                logger.info(f"anomaly_map_batch shape {anomaly_map_batch.shape}")

        except Exception as e:
            logger.error(f"Error during anomaly map calculation for {run_context}: {e}. Skipping batch.")
            continue # Skip this batch

        # --- Store results for the Evaluator ---
        # Ensure shapes match expectations (N, H, W) for maps/masks, (N,) for labels
        all_anomaly_maps.extend(anomaly_map_batch) # Add batch of maps (numpy arrays)
        all_gt_masks.extend(mask.squeeze(1).cpu().numpy()) # Remove channel dim, move to CPU
        all_gt_labels.extend(label.cpu().numpy())
        # Store the actual class names from the batch
        all_class_names.extend(batch_class_names)

    # --- Prepare data for Evaluator ---
    if not all_anomaly_maps:
        logger.error(f"No anomaly maps successfully generated for context {run_context}. Cannot evaluate.")
        return {}
    logger.info(f"Evaluating context: results_dict...")
    # class_name_list_np = np.array(all_class_names)
    results_dict = {
        'imgs_masks': np.stack(all_gt_masks, axis=0).astype(np.uint8), # Stack to (N, H, W)
        'anomaly_maps': np.stack(all_anomaly_maps, axis=0),     # Stack to (N, H, W)
        'anomalys': np.stack(all_gt_labels, axis=0),             # Stack to (N,)
        'cls_names': np.array(all_class_names)                   # Array of actual class names (N,)
    }

    ## Important:
    ##-------------
    ## Depending on run_context, we will trigger different routine to compute the metrics.
    ## It is kind of not the most efficient way, maybe we will need to revise later on.
    if run_context == FULL_DATASET_IDENTIFIER:
        metrics = []
        for cls_name in CLASS_NAMES:
            metrics_cls_name = evaluator.run(results_dict, cls_name)
            metrics.append(metrics_cls_name)
    else:
        metrics = evaluator.run(results_dict, run_context) # run_context provides overall context

    # --- Save Debug Images ---
    if debug_image_indices and test_imgs_for_debug:
        logger.info(f"Saving debug images for {run_context} to {save_dir}...")
        # Normalize anomaly maps for consistent visualization
        viz_anomaly_maps = results_dict['anomaly_maps']
        map_min, map_max = viz_anomaly_maps.min(), viz_anomaly_maps.max()
        if map_max > map_min:
             viz_anomaly_maps_norm = (viz_anomaly_maps - map_min) / (map_max - map_min + 1e-6) # Add epsilon
        else:
             viz_anomaly_maps_norm = np.zeros_like(viz_anomaly_maps)

        save_debug_images(
            indices=debug_image_indices,
            test_imgs=test_imgs_for_debug, # List of original images (C, H, W) numpy
            all_class_names=results_dict['cls_names'], # Pass all class names (N,)
            epoch_number=epoch_str,
            anomaly_maps=viz_anomaly_maps_norm, # Use normalized maps for viz (N, H, W)
            gt_masks=results_dict['imgs_masks'], # Ground truth masks (N, H, W)
            mean=mean_train,
            std=std_train,
            save_dir=save_dir,
            run_context=run_context, # Pass overall context
            model_name=args.model
        )


    return metrics


def save_debug_images(
    indices: List[int],
    test_imgs: List[np.ndarray], # List of (C, H, W) numpy arrays
    all_class_names: np.ndarray, # Array of class names for all test images (N,)
    epoch_number: Union[int, str, None],
    anomaly_maps: np.ndarray, # All anomaly maps (N, H, W) - expected normalized [0,1] for viz
    gt_masks: np.ndarray,   # All ground truth masks (N, H, W)
    mean: List[float],
    std: List[float],
    save_dir: Path,
    run_context: str, # e.g., 'bottle' or 'full'
    model_name: str
):
    """Saves comparison images for debugging.

    Args:
        indices:
        test_imgs:
        all_class_names:
        epoch_number:
        anomaly_maps:
        gt_masks:
        mean:
        std:
        save_dir:
        run_context:
        model_name:
    """
    num_images = len(test_imgs)
    if num_images == 0: return
    save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    plot_cols = 3 # Original, Anomaly Map, GT Mask
    fig_width = 12
    epoch_str = str(epoch_number) if epoch_number is not None else 'final'

    for idx in indices:
        if idx < 0 or idx >= num_images:
            logger.warning(f"Debug image index {idx} out of bounds (0-{num_images-1}). Skipping.")
            continue

        # Get data for the specific index
        img_orig_chw = test_imgs[idx]
        score_map = anomaly_maps[idx]
        gt_mask = gt_masks[idx]
        actual_class_name = all_class_names[idx] # Get the class name for this specific image

        # Denormalize original image (CHW -> HWC for display)
        img_orig_vis = denormalization(img_orig_chw, mean, std) # Assumes returns HWC

        fig, axes = plt.subplots(1, plot_cols, figsize=(fig_width, 4))
        # Include actual class name in title
        fig.suptitle(f"Debug Idx:{idx} - Actual Class: {actual_class_name} (Context: {run_context}) Epoch: {epoch_str}")

        axes[0].imshow(img_orig_vis)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        im = axes[1].imshow(score_map, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Anomaly Map")
        axes[1].axis('off')

        axes[2].imshow(gt_mask, cmap='gray')
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout for sub title
        # Include actual class name in filename for clarity, especially for 'full' context
        save_path = save_dir / f"debug_{model_name}_{run_context}_idx{idx}_{actual_class_name}_epoch_{epoch_str}.png"

        try:
            plt.savefig(save_path)
        except Exception as e:
            logger.error(f"Failed to save debug image {save_path}: {e}")
        plt.close(fig)
        ## save image ==== improved
        img_orig_chw_tensor = torch.from_numpy(img_orig_chw)
        mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=img_orig_chw_tensor.device)
        std_tensor = torch.tensor([0.229, 0.224, 0.225], device=img_orig_chw_tensor.device)
        img_rec = img_orig_chw_tensor * std_tensor[:, None, None] + mean_tensor[:, None, None]
        img_rec = torch.clamp(img_rec, 0, 1)
        # RGB image
        img_rec = Image.fromarray((img_rec * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0))
        anomaly_map = np.squeeze(score_map)
        anomaly_map = anomaly_map / anomaly_map.max()
        anomaly_map = cm.jet(anomaly_map)
        anomaly_map = (anomaly_map[:, :, :3] * 255).astype('uint8')
        anomaly_map = Image.fromarray(anomaly_map)  # bug here
        img_rec_anomaly_map = Image.blend(img_rec, anomaly_map, alpha=0.4)
        # mask
        #img_mask = Image.fromarray((gt_mask * 255).astype(np.uint8).transpose(1, 2, 0).repeat(3, axis=2))
        # mask
        # Ensure gt_mask is numpy and handle potential tensor input
        if isinstance(gt_mask, torch.Tensor):
             gt_mask = gt_mask.cpu().numpy()
        # Ensure gt_mask is suitable type and shape for conversion
        # Assuming gt_mask is (1, H, W) or (H, W), needs to become (H, W, 1) for transpose then repeat
        if gt_mask.ndim == 3 and gt_mask.shape[0] == 1: # If (1, H, W)
            gt_mask = gt_mask.squeeze(0) # Becomes (H, W)
        if gt_mask.ndim == 2: # If (H, W)
            gt_mask = gt_mask[:, :, np.newaxis] # Becomes (H, W, 1)

        # Ensure mask is binary 0 or 1 before scaling
        gt_mask = (gt_mask > 0.5).astype(np.uint8) # Example thresholding

        img_mask = Image.fromarray((gt_mask * 255).repeat(3, axis=2))
        # Save in figure in a row
        fig, axes = plt.subplots(1, plot_cols, figsize=(fig_width, 4))
        fig.suptitle(f"Debug Idx:{idx} - Actual Class: {actual_class_name} (Context: {run_context}) Epoch: {epoch_str}")

        axes[0].imshow(img_rec)
        axes[0].set_title("img_rec")
        axes[0].axis('off')

        axes[1].imshow(img_rec_anomaly_map)
        axes[1].set_title("img_rec_anomaly_map")
        axes[1].axis('off')

        axes[2].imshow(img_mask, cmap='gray')
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
        save_path = save_dir / f"debug_{model_name}_{run_context}_idx{idx}_{actual_class_name}_epoch_{epoch_str}_vis2.png"
        plt.savefig(save_path)
        plt.close(fig)




# Function to run experiment for a SINGLE class
def run_experiment(
    args: argparse.Namespace,
    class_name: str, # Specific class name
    base_save_dir: Path,
    device: torch.device,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
) -> Dict[str, Any]:
    """Runs the full training and evaluation pipeline for a single class.

    Args:
        args:
        class_name:
        base_save_dir:
        device:
        wandb_run:
    """

    logger.info("\n" + "="*50)
    logger.info(f" Starting Experiment for Class: {class_name} ")
    logger.info("="*50 + "\n")

    class_args = copy.deepcopy(args) # Make a copy if needed, but often not required with sweeps
    class_args.obj = class_name # Set the class name in args for dataloader
    set_seed(class_args.seed) # Set seed for reproducibility for this specific run

    start_time = time.time()

    # --- Setup for the specific class ---
    # args.obj = class_name # Set the class name in args (already done in main loop)
    set_seed(args.seed) # Set seed for reproducibility for this specific run

    # Create specific save directory for this class run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    current_save_dir = base_save_dir / f"{class_name}_{class_args.model}_{run_timestamp}"
    current_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results for class '{class_name}' to: {current_save_dir}")

    # --- Data Loaders ---
    logger.info(f"Loading datasets for class: {class_name}...")
    try:
        # Use get_loader for single class
        class_args.obj = class_name
        train_loader, val_loader, test_loader = get_dataloader(class_args)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        logger.info(f"Approx Train samples: {len(train_loader.dataset)}, Approx Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Error getting dataloaders for class {class_name}: {e}", exc_info=True)
        return {"class": class_name, "status": "failed", "error": f"Dataloader error: {e}"}

    # --- Model, Optimizer, Criterion, Scaler ---
    model = get_model(args=class_args, device=device)
    optimizer = get_optimizer(args=class_args, model=model)
    criterion = get_criterion(args=class_args).to(device)
    # if args.loss_func == "L2Loss":
    #     criterion = L2Loss().to(device) # Ensure loss is on the correct device
    # elif args.loss_func == "CosLoss":
    #     criterion = CosLoss().to(device)
    # elif args.loss_func == "KLLoss":
    #     criterion = KLLoss().to(device)
    # else:
    #     logger.info(f"{args.loss_func} was not supported yet, using default L2Loss")
    #     args.loss_func = "L2Loss"
    #     criterion = L2Loss().to(device) # Ensure loss is on the correct device

    logger.info(f"Loss function: {class_args.loss_func}")

    scaler = amp.GradScaler(enabled=class_args.amp)

    # --- Instantiate Evaluator ---
    evaluator = Evaluator(
        metrics=METRICS_TO_COMPUTE,
        pooling_ks=None, # Adjust if needed
        max_step_aupro=100, # Default
        mp=False, # Set to True to try multiprocessing for AUPRO
        use_adeval=class_args.use_adeval if hasattr(class_args, 'use_adeval') else False
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = current_save_dir / f"model_{class_name}_{class_args.model}_best.pt"
    all_metrics = {} # To store metrics from the last evaluation

    logger.info(f"Starting training for {class_args.epochs} epochs...")
    for epoch in range(1, class_args.epochs + 1):
        train_loss = train_epoch(class_args, model, train_loader, optimizer, criterion, scaler, device, epoch, run_context=class_name)

        # Validation and potential early stopping
        if epoch % class_args.val_epochs == 0 or epoch == class_args.epochs: # Validate on schedule or last epoch
            val_loss = validate_epoch(class_args, model, val_loader, criterion, device, epoch, run_context=class_name)
            logger.info(f"Epoch {epoch}/{class_args.epochs} => Class: {class_name}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            log_dict = {
                f"{class_name}/train_loss": train_loss,
                f"{class_name}/val_loss": val_loss,
                "epoch": epoch # Log epoch globally
            }

            if epoch % class_args.val_epochs == 0 or epoch == class_args.epochs:
                 logger.info(f"Running evaluation at epoch {epoch}...")
                 all_metrics = evaluate_performance(
                     args=class_args,
                     model=model,
                     test_loader=test_loader,
                     epoch_number=epoch,
                     device=device,
                     save_dir=current_save_dir / "eval_debug_intermediate", # Subdir for debug images
                     run_context=class_name, # Pass class name as context
                     evaluator=evaluator,
                     debug_image_indices=DEFAULT_IMAGE_TO_VISUAL,
                 )
                 # Log intermediate evaluation metrics to WandB
                 if wandb_run and all_metrics:
                     for m_name, m_val in all_metrics.items():
                         # Prefix with class name and 'eval'
                         log_dict[f"{class_name}/eval_{m_name}"] = m_val

            # Log metrics to WandB (if enabled)
            if wandb_run:
                 # wandb.log(log_dict, step=epoch) # Log with epoch step
                 wandb.log(log_dict) # Log with epoch step

        else:
            # Log only training loss if not a validation epoch
             log_dict = { f"{class_name}/train_loss": train_loss, "epoch": epoch }
             if wandb_run:
                 # wandb.log(log_dict, step=epoch)
                 wandb.log(log_dict)


    logger.info(f"Training finished for class {class_name}.")

    # --- Final Model Saving ---
    final_model_path = current_save_dir / f"model_{class_name}_{class_args.model}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # --- Final Evaluation ---
    # Load the best model if it exists, otherwise use the final model
    load_path = final_model_path
    if best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path} for final evaluation...")
        load_path = best_model_path
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}. Using final model for evaluation.")

    # Re-instantiate model structure and load state dict
    eval_model = get_model(args=class_args, device=device) # Get a fresh instance
    try:
        eval_model.load_state_dict(torch.load(load_path, map_location=device))
        logger.info(f"Successfully loaded model weights from {load_path}")
    except Exception as e:
        logger.error(f"Failed to load model state dict from {load_path}: {e}")
        return {"class": class_name, "status": "failed", "error": f"Model loading failed: {e}"}

    logger.info(f"\n--- Running Final Evaluation for Class: {class_name} ---")

    final_eval_metrics = evaluate_performance(
        args=class_args,
        model=eval_model, # Use the loaded model
        test_loader=test_loader,
        epoch_number='final', # Indicate final evaluation
        device=device,
        save_dir=current_save_dir / "eval_debug_final", # Subdir for final debug images
        run_context=class_name, # Pass class name as context
        evaluator=evaluator, # Re-use or re-create evaluator if needed
        debug_image_indices = DEFAULT_IMAGE_TO_VISUAL
    )

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Experiment for class {class_name} finished in {duration:.2f} seconds.")

    # --- Results ---
    results = {
        "class": class_name,
        "status": "completed" if final_eval_metrics else "evaluation_failed",
        "metrics": final_eval_metrics, # Store the dict of metrics from final eval
        "training_time_seconds": duration,
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path) if best_model_path.exists() else "N/A",
        "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
        "save_dir": str(current_save_dir)
    }

    # Log final metrics for this class to WandB Summary
    if wandb_run and final_eval_metrics:
        for metric_name, metric_value in final_eval_metrics.items():
            # Prefix with class name for summary clarity when running multiple classes
            wandb.summary[f"{class_name}/{metric_name}"] = metric_value
        wandb.summary[f"{class_name}/training_time_seconds"] = duration
        wandb.summary[f"{class_name}/best_val_loss"] = results["best_val_loss"]

    if wandb_run and class_args.log_images and (current_save_dir / "eval_debug_final").exists():
        try:
            debug_img_folder = current_save_dir / "eval_debug_final"
            img_paths = list(debug_img_folder.glob("debug_*.png"))
            if img_paths:
                wandb.log({f"{class_name}/debug_images": [wandb.Image(str(p)) for p in img_paths]})
                logger.info(f"Logged {len(img_paths)} debug images to WandB.")
        except Exception as e:
            logger.error(f"Failed to log debug images to WandB: {e}")

    return results


def run_experiment_all(
    args: argparse.Namespace,
    base_save_dir: Path,
    device: torch.device,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    list_categories: Optional[List[str]] = None # Allow specifying categories, default to all
) -> Union[dict[str, str], dict[str, str], list[dict[str, Union[Union[str, float, dict[str, float], None], Any]]]]:
    """Runs the full training and evaluation pipeline on the combined dataset.

    Args:
        args:
        base_save_dir:
        device:
        wandb_run:
        list_categories:
    """

    context_name = FULL_DATASET_IDENTIFIER # Use the identifier
    logger.info("\n" + "="*50)
    logger.info(f" Starting Experiment for Full Dataset ({context_name}) ")
    logger.info("="*50 + "\n")

    start_time = time.time()

    # --- Setup for the full dataset run ---
    class_args = copy.deepcopy(args)
    set_seed(class_args.seed) # Set seed for reproducibility for this specific run


    # Create specific save directory for this full run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Use the context_name in the directory
    current_save_dir = base_save_dir / f"{context_name}_{class_args.model}_{run_timestamp}"
    current_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results for '{context_name}' run to: {current_save_dir}")

    # --- Data Loaders ---
    logger.info(f"Loading combined dataset for categories: {'ALL' if list_categories is None else list_categories}...")
    try:
        # Use get_loader_full for the combined dataset
        train_loader, val_loader, test_loader = get_loader_full(class_args, list_categories=list_categories)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        # Note: len(loader.dataset) on ConcatDataset gives total size
        logger.info(f"Total Train samples: {len(train_loader.dataset)}, Total Val samples: {len(val_loader.dataset)}, Total Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Error getting combined dataloaders: {e}", exc_info=True)
        return {"class": context_name, "status": "failed", "error": f"Dataloader error: {e}"}

    # --- Model, Optimizer, Criterion, Scaler ---
    model = get_model(args= class_args, device=device)
    optimizer = get_optimizer(args=class_args, model=model)
    criterion = get_criterion(args=class_args).to(device=device)
    #criterion = L2Loss().to(device)
    # if args.loss_func == "L2Loss":
    #     criterion = L2Loss().to(device) # Ensure loss is on the correct device
    # elif args.loss_func == "CosLoss":
    #     criterion = CosLoss().to(device)
    # elif args.loss_func == "KLLoss":
    #     criterion = KLLoss().to(device)
    # else:
    #     logger.info(f"{args.loss_func} was not supported yet, using default L2Loss")
    #     args.loss_func =  "L2Loss"
    #     criterion = L2Loss().to(device) # Ensure loss is on the correct device

    logger.info(f"Loss function: {class_args.loss_func}")
    scaler = amp.GradScaler(enabled=class_args.amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=class_args.epochs, eta_min=1e-6)
    # --- Instantiate Evaluator ---
    evaluator = Evaluator(
        metrics=METRICS_TO_COMPUTE,
        pooling_ks=None,
        max_step_aupro=100,
        mp=False,
        use_adeval=class_args.use_adeval if hasattr(class_args, 'use_adeval') else False
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = current_save_dir / f"model_{context_name}_{class_args.model}_best.pt"
    all_metrics = {} # To store metrics from the last evaluation

    logger.info(f"Starting training on combined dataset for {class_args.epochs} epochs...")
    for epoch in range(1, class_args.epochs + 1):
        # Pass context_name to training functions
        train_loss = train_epoch(class_args, model, train_loader, optimizer, criterion, scaler, device, epoch, run_context=context_name)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # if wandb_run: wandb.log({f"{class_name}/learning_rate": current_lr, "epoch": epoch})
        # Validation and potential early stopping
        if epoch % class_args.val_epochs == 0 or epoch == class_args.epochs:
            val_loss = validate_epoch(class_args, model, val_loader, criterion, device, epoch, run_context=context_name)
            logger.info(f"Epoch {epoch}/{class_args.epochs} LR {current_lr} => Context: {context_name}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            # Use simplified keys for full dataset run in WandB logs
            log_dict = {
                f"{context_name}/train_loss": train_loss,
                f"{context_name}/val_loss": val_loss,
                f"{context_name}/learning_rate": current_lr,
                "epoch": epoch
            }

            logger.info(f"\n--- Running Evaluation for Full Dataset ({context_name}) at {epoch}---")
            current_args_obj = class_args.obj
            for cls_name in list_categories:
                class_args.obj = cls_name
                _, _, test_data_loader = get_dataloader(class_args)
                eval_metrics = evaluate_performance(
                    args=class_args,
                    model=model,
                    test_loader=test_data_loader,
                    epoch_number=epoch,
                    device=device,
                    save_dir=current_save_dir / "eval_debug_final",
                    run_context=cls_name,  # Pass 'full' context
                    evaluator=evaluator,
                    debug_image_indices=DEFAULT_IMAGE_TO_VISUAL
                )
                for metric_name, metric_value in eval_metrics.items():
                    log_dict[f"{cls_name}/{metric_name}"] = metric_value

            # return args.obj
            class_args.obj = current_args_obj
            if wandb_run:
                wandb.log(log_dict, step=epoch)

            ## Early stop is not employed for now
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     epochs_without_improvement = 0
            #     torch.save(model.state_dict(), best_model_path)
            #     logger.info(f"Validation loss improved to {val_loss:.6f}. Saved best model to {best_model_path}")
            # else:
            #     epochs_without_improvement += args.val_epochs

            # if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            #     logger.info(f"Early stopping triggered for {context_name} run after {epoch} epochs.")
            #     break
        else:
             # Log only training loss if not a validation epoch
             log_dict = { f"{context_name}/train_loss": train_loss,
                          f"{context_name}/learning_rate": current_lr,
                          "epoch": epoch }
             if wandb_run:
                 wandb.log(log_dict, step=epoch)

    logger.info(f"Training finished for {context_name} run.")

    # --- Final Model Saving ---
    final_model_path = current_save_dir / f"model_{context_name}_{class_args.model}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model for {context_name} run to {final_model_path}")

    # --- Final Evaluation ---
    load_path = final_model_path
    if best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path} for final evaluation...")
        load_path = best_model_path
    else:
        logger.warning(f"Best model checkpoint not found. Using final model for {context_name} evaluation.")

    eval_model = get_model(args=class_args, device=device) # Fresh instance
    try:
        eval_model.load_state_dict(torch.load(load_path, map_location=device))
        logger.info(f"Successfully loaded model weights from {load_path}")
    except Exception as e:
        logger.error(f"Failed to load model state dict from {load_path}: {e}")
        return {"class": context_name, "status": "failed", "error": f"Model loading failed: {e}"}

    logger.info(f"\n--- Running Final Evaluation for Full Dataset ({context_name}) ---")
    results = []
    for cls_name in list_categories:
        class_args.obj = cls_name
        _,_, test_data_loader = get_dataloader(class_args)
        eval_metrics = evaluate_performance(
            args=class_args,
            model=eval_model,
            test_loader=test_data_loader,
            epoch_number='final',
            device=device,
            save_dir=current_save_dir / "eval_debug_final",
            run_context=cls_name, # Pass 'full' context
            evaluator=evaluator,
            debug_image_indices=DEFAULT_IMAGE_TO_VISUAL
        )
        duration = time.time() - start_time

        cls_results = {
            "class": cls_name,
            "status": "completed" if eval_metrics else "evaluation_failed",
            "metrics": eval_metrics,  # This will contain aggregated and per-class metrics
            "training_time_seconds": duration,
            "final_model_path": str(final_model_path),
            "best_model_path": str(best_model_path) if best_model_path.exists() else "N/A",
            "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
            "save_dir": str(current_save_dir)
        }
        if wandb_run and eval_metrics:
            for metric_name, metric_value in eval_metrics.items():
                # Use context_name prefix for summary clarity
                wandb.summary[f"{cls_name}/{metric_name}"] = metric_value
            wandb.summary[f"{cls_name}/training_time_seconds"] = duration
            wandb.summary[f"{cls_name}/best_val_loss"] = cls_results["best_val_loss"]
        results.append(cls_results)

        if wandb_run and class_args.log_images and (current_save_dir / "eval_debug_final").exists():
            try:
                debug_img_folder = current_save_dir / "eval_debug_final"
                img_paths = list(debug_img_folder.glob("debug_*.png"))
                if img_paths:
                    wandb.log({f"{cls_name}/debug_images": [wandb.Image(str(p)) for p in img_paths]})
                    logger.info(f"Logged {len(img_paths)} debug images to WandB.")
            except Exception as e:
                logger.error(f"Failed to log debug images to WandB: {e}")


    # logger.info(results)
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Experiment for {context_name} run finished in {duration:.2f} seconds.")

    return results

def main():
    start_time_full_script = time.time()
    # --- Global Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    # wandb sweep setup
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="MVTec AD Training and Evaluation with ViTAD")
    # Add arguments based on DEFAULT_CONFIG
    for key, value in DEFAULT_CONFIG.items():
        arg_type = type(value)
        if isinstance(value, bool):
            # Special handling for boolean flags
             parser.add_argument(f'--{key}', action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, list):
             parser.add_argument(f'--{key}', type=type(value[0]) if value else str, nargs='+', default=value)
        else:
            # Handle None default specifically if needed, otherwise standard type
            parser.add_argument(f'--{key}', type=arg_type if value is not None else str, default=value)

    # Parse arguments BEFORE initializing wandb
    args = parser.parse_args()
    args_save_dir = Path(args.save_dir)
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Base directory for all runs initiated by this script execution
    base_save_dir = Path(
        args_save_dir) / f"run_{run_timestamp}_{args.model}_{args.obj}_{args.loss_func}_{args.epochs}"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Base save directory for this run: {base_save_dir}")
    # --- Wandb Initialization ---
    wandb_run = None
    # Determine wandb run name based on mode
    wandb_mode = "disabled"
    if args.wandb_log and (not hasattr(args, 'mode') or args.mode.lower() != 'disabled'):
        wandb_mode = "online" # Default to online if enabled
        if hasattr(args, 'mode') and args.mode: # Check if mode is specified
            wandb_mode = args.mode

    if wandb_mode != "disabled":
        try:
            # Adjust run name based on whether it's single, all, or full
            run_identifier = args.obj if args.obj else 'unknown'
            run_name = f"{args.model}_{run_identifier}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(
                project=args.project,
                name=run_name,
                tags=args.tags + [args.model, args.obj if args.obj else 'N/A'],
                notes=f"{args.notes} _ model {args.model} loss_func {args.loss_func}",
                config=vars(args), # Log all arguments
                dir=str(base_save_dir), # Set wandb log directory within the run's base dir
                mode=wandb_mode, # online, offline, or disabled
                settings=wandb.Settings(start_method="fork"), # Use fork for better compatibility
                reinit=True,
            )
            logger.info(f"Wandb initialized. Mode: '{wandb_mode}'. Run name: {run_name}")

            # --- IMPORTANT: Update args with sweep config ---
            for key, value in wandb.config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    logger.warning(f"Sweep config key '{key}' not found in script args, adding dynamically.")
                    setattr(args, key, value)  # Or handle as needed

            logger.info(f"Args updated with W&B sweep config: {vars(args)}")

        except Exception as e:
            logger.error(f"Error initializing Wandb: {e}. Wandb logging disabled for this execution.")
            wandb_run = None # Ensure it's None if init fails
    else:
        logger.info("Wandb logging is disabled (wandb_log=False or mode='disabled').")

    # --- Determine Experiment Mode ---
    all_results = []
    experiment_mode = "unknown"
    classes_to_run = CLASS_NAMES

    if args.obj == 'all': # Run training for all classes separately
        classes_to_run = CLASS_NAMES
        logger.info(f"Experiment Mode: Running separate experiments for ALL MVTec AD classes: {classes_to_run}")
        experiment_mode = "all_separate"
        try:
            for class_name in classes_to_run:
                args.obj = class_name

                class_results = run_experiment(
                    args=args,
                    class_name=class_name,
                    base_save_dir=base_save_dir, # Each run gets its own subfolder inside this
                    device=device,
                    wandb_run=wandb_run # Pass the single wandb run object

                )
                args.obj='all' # need to restore
                all_results.append(class_results)
        except Exception as e:
             logger.error(f"An unexpected error occurred during the 'all_separate' execution loop: {e}", exc_info=True)

    elif args.obj == FULL_DATASET_IDENTIFIER: # Run training on all classes as a whole dataset, e.g in one run
        logger.info(f"Experiment Mode: Running ONE experiment on the FULL combined MVTec AD dataset.")
        experiment_mode = "full_combined"
        try:
            full_results = run_experiment_all(
                 args=args,
                 base_save_dir=base_save_dir,
                 device=device,
                 wandb_run=wandb_run,
                 list_categories=CLASS_NAMES # Use all classes from CLASS_NAMES by default
            )
            all_results = full_results
        except Exception as e:
            logger.error(f"An unexpected error occurred during the 'full_combined' execution: {e}", exc_info=True)

    elif args.obj in CLASS_NAMES: # Run for a single specified class
        class_name = args.obj
        classes_to_run = class_name
        logger.info(f"Experiment Mode: Running experiment for a SINGLE class: {class_name}")
        experiment_mode = "single_class"
        try:
             # args.obj is already set correctly
             single_class_results = run_experiment(
                 args=args,
                 class_name=class_name,
                 base_save_dir=base_save_dir,
                 device=device,
                 wandb_run=wandb_run
             )
             all_results.append(single_class_results)
        except Exception as e:
            logger.error(f"An unexpected error occurred during the 'single_class' execution for {class_name}: {e}", exc_info=True)
    else:
        logger.error(f"Error: Invalid value for --obj argument: '{args.obj}'.")
        logger.error(f"Choose from: {CLASS_NAMES}, 'all', or '{FULL_DATASET_IDENTIFIER}'.")
        if wandb_run: wandb.finish(exit_code=1) # Exit wandb run with error code
        sys.exit(1)

    # --- Final Summary ---
    logger.info("\n" + "="*80)
    logger.info(f" Overall Results Summary (Mode: {experiment_mode})")
    logger.info("="*80)

    if not all_results:
        logger.warning("No results were generated.")
    else:
        summary_metrics = METRICS_TO_COMPUTE
        header = f"{'Context/Class':<15} | {'Status':<10} | " + " | ".join([f'{m:<15}' for m in summary_metrics]) + f" | {'Time (s)':<10}"
        logger.info(header)
        logger.info("-" * len(header))

        # Aggregated metrics (only relevant for 'all_separate' mode)
        aggregated_metrics = {m: [] for m in summary_metrics}
        completed_count = 0
        total_time = 0.0

        for result in all_results:
            status = result['status']
            context = result['class']
            time_s = result.get('training_time_seconds', 0)

            if status == 'completed':
                metrics = result.get('metrics', {}) # Metrics dict from evaluation
                metric_values = [metrics.get(m, np.nan) for m in summary_metrics] # Get values, use NaN if missing

                # Format metric values for display
                metric_strings = []
                for v in metric_values:
                    if isinstance(v, (float, np.floating)) and not np.isnan(v):
                         metric_strings.append(f'{v:.4f}')
                    else:
                         metric_strings.append('N/A') # Handle NaN or missing

                row = f"{context:<15} | {status:<10} | " + " | ".join([f'{s:<15}' for s in metric_strings]) + f" | {time_s:<10.2f}"
                logger.info(row)

                # Add to aggregation only if running separate classes
                for m, v in zip(summary_metrics, metric_values):
                    if not np.isnan(v):
                        aggregated_metrics[m].append(v)
                completed_count += 1
                total_time += time_s
            else:
                error_msg = result.get('error', 'Unknown')
                row = f"{context:<15} | {status:<10} | " + " | ".join(['N/A'.center(15)] * len(summary_metrics)) + f" | {'N/A':<10}"
                logger.info(row)
                logger.warning(f"  -> Failed Run '{context}' Reason: {error_msg}")

        # Display average only if multiple separate classes were run successfully
        #if experiment_mode == "all_separate" and completed_count > 0:
        if completed_count > 0:
             avg_metrics = {m: np.mean(vals) for m, vals in aggregated_metrics.items() if vals}
             avg_metric_values = [avg_metrics.get(m, np.nan) for m in summary_metrics]
             avg_time = total_time / completed_count

             avg_metric_strings = []
             for v in avg_metric_values:
                 if not np.isnan(v): avg_metric_strings.append(f'{v:.4f}')
                 else: avg_metric_strings.append('N/A')

             logger.info("-" * len(header))
             avg_row = f"{'Average':<15} | {'':<10} | " + " | ".join([f'{s:<15}' for s in avg_metric_strings]) + f" | {avg_time:<10.2f}"
             logger.info(avg_row)
             logger.info("=" * len(header))

             # Log average metrics to WandB Summary if running all separate
             if wandb_run:
                 for m, v in avg_metrics.items():
                     wandb.summary[f"average/{m}"] = v
                 wandb.summary["average/training_time_seconds"] = avg_time
                 wandb.summary["completed_classes"] = completed_count
                 wandb.summary["total_classes_run"] = len(classes_to_run)
                 primary_metric_key = PRIMARY_KEY_METRIC
                 if primary_metric_key in wandb.summary.keys():
                     wandb.log({"sweep_metric": wandb.summary[primary_metric_key]})
                     logger.info(
                         f"Logged primary sweep metric '{primary_metric_key}': {wandb.summary[primary_metric_key]:.4f}")
                 else:
                     logger.warning(f"Primary sweep metric key '{primary_metric_key}' not found in summary.")

        elif experiment_mode == "all_separate" and completed_count == 0:
             logger.info("\nNo experiments completed successfully in 'all_separate' mode.")

    total_duration_script = time.time() - start_time_full_script
    logger.info(f"\nTotal script execution time: {total_duration_script:.2f} seconds ({total_duration_script/60:.2f} minutes).")
    if wandb_run:
        wandb.summary["total_script_duration_seconds"] = total_duration_script
        # Finish Wandb Run
        wandb.finish()
        logger.info("Wandb run finished.")

if __name__ == "__main__":
    main()
