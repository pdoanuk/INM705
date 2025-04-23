# -*- coding: utf-8 -*-
"""
MVTec AD Anomaly Detection Training and Evaluation Pipeline
Using Vision Transformer (ViT) Encoder-Decoder Variants.

TIMESTAMP @ 2025-04-10T23:45:47
author: phuocddat
update losses
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
import wandb

from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from typing import Tuple, List, Dict, Any, Optional
from losses import L2Loss, L1Loss, CosLoss


try:
    from dataset_mvtec import get_dataloader

    # from model_vit import ViTAutoencoder
    from model_vit import VitDecoderExp
    from utils_mvtec import set_seed, denormalization, log_write
    from config import args, CLASS_NAMES, mean_train, std_train
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

SIGMA_DETECTION = 15    # Sigma for Gaussian filter for detection scores
SIGMA_SEGMENTATION = 6  # Sigma for Gaussian filter for segmentation scores
WANDB_PROJECT_NAME = "INM705-exp" #
WANDB_TAGS = ["experiment", "ViT-AE"] #

# %% Helper Functions

def get_model(device: torch.device) -> nn.Module:
    """Instantiates and returns the anomaly detection model."""
    # Configure this function to return the desired model
    # model = ViTAutoencoder().to(device)
    model = VitDecoderExp().to(device)
    print(f"Using model: {model.__class__.__name__}")
    try:
        # Debug model summary
        summary(model, input_size=(args.batch_size, 3, args.image_size, args.image_size))
    except Exception as e:
        print(f"Could not generate model summary: {e}")
    return model

def get_optimizer(model: nn.Module) -> optim.Optimizer:
    """Instantiates and returns the optimizer."""
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)

    # optimizer = optim.Adam(
    #     params=model.parameters(),
    #     lr=args.lr,
    #     betas=(args.beta1, args.beta2)
    # )
    print(f"Using optimizer: {optimizer.__class__.__name__} with LR: {args.lr}")
    return optimizer


def train_epoch(
    args: Any,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: amp.GradScaler,
    device: torch.device,
    epoch: int
) -> float:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

    #for x, _, _ in pbar:
    for x, y, mask, cls_name in pbar:
        x = x.to(device)
        optimizer.zero_grad()

        if args.amp:
            with amp.autocast():
                x_hat = model(x)
                loss = criterion(x_hat, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{args.epochs} [Train] Avg Loss: {avg_loss:.6f}")
    return avg_loss

def validate_epoch(
    args: Any,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """Runs a single validation epoch."""
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [Val]")

    with torch.no_grad():
        #for x, _, _ in pbar:
        for x, y, mask, cls_name in pbar:
            x = x.to(device)
            if args.amp:
                with amp.autocast():
                    x_hat = model(x)
                    loss = criterion(x_hat, x)
            else:
                x_hat = model(x)
                loss = criterion(x_hat, x)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{args.epochs} [Val] Avg Loss: {avg_loss:.6f}")
    return avg_loss


def evaluate_performance(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    save_dir: Path,
    class_name: str,
    debug_image_indices: Optional[List[int]] = [5, 10, 15] # Indices to save debug images for
) -> Tuple[float, float]:
    """Runs the full test evaluation pipeline."""
    model.eval()
    mse_loss_func = nn.MSELoss(reduction='none') # Per-pixel MSE for anomaly maps
    # mse_loss_func = L2Loss(reduction='none')
    #mse_loss_func = CosLoss()

    det_scores, seg_scores = [], []
    test_imgs_list, gt_list, gt_mask_list, recon_imgs_list = [], [], [], []

    print(f"Running evaluation for class: {class_name}...")
    pbar = tqdm(test_loader, desc=f"Evaluation [{class_name}]")

    #for (x, label, mask) in pbar:
    for (x, label, mask, cls_name) in pbar:
        test_imgs_list.extend(x.cpu().numpy())
        gt_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.squeeze(0).cpu().numpy())
        x = x.to(device)
        with torch.no_grad():
            if args.amp:
                 with amp.autocast():
                    x_hat = model(x)
                    mse_per_pixel = mse_loss_func(x_hat, x)
            else:
                x_hat = model(x)
                mse_per_pixel = mse_loss_func(x_hat, x) # Shape: (B, C, H, W)

        # Anomaly score map (mean over channels)
        anomaly_map = mse_per_pixel.mean(dim=1).cpu().numpy() # Shape: (B, H, W)

        # Apply Gaussian Filter for detection and segmentation scores
        batch_det_scores = copy.deepcopy(anomaly_map)
        batch_seg_scores = copy.deepcopy(anomaly_map)

        for i in range(anomaly_map.shape[0]):
            # Detection score: Smoothed anomaly map
            batch_det_scores[i] = gaussian_filter(batch_det_scores[i], sigma=SIGMA_DETECTION)
            # Segmentation score map: Smoothed anomaly map
            batch_seg_scores[i] = gaussian_filter(batch_seg_scores[i], sigma=SIGMA_SEGMENTATION)

        det_scores.extend(batch_det_scores)
        seg_scores.extend(batch_seg_scores)
        recon_imgs_list.extend(x_hat.cpu().numpy())


    # Segmentation Score Normalization and AUC calculation
    seg_scores_np = np.asarray(seg_scores) # Shape: (N, H, W)
    max_anomaly_score = seg_scores_np.max()
    min_anomaly_score = seg_scores_np.min()
    # Normalize scores to [0, 1] range for consistent thresholding/AUC
    if max_anomaly_score > min_anomaly_score:
        seg_scores_normalized = (seg_scores_np - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    else:
        print("Warning: Max and Min anomaly scores are equal. Setting normalized scores to 0.")
        seg_scores_normalized = np.zeros_like(seg_scores_np)

    gt_mask_np = np.asarray(gt_mask_list).astype(int) # Shape: (N, H, W)

    # Calculate pixel-level ROC AUC
    try:
        per_pixel_rocauc = roc_auc_score(gt_mask_np.flatten(), seg_scores_normalized.flatten())
        print(f"Pixel-wise ROC AUC: {per_pixel_rocauc:.4f}")
    except ValueError as e:
        print(f"Warning: Could not calculate pixel ROC AUC. Maybe only normal samples in test? Error: {e}")
        per_pixel_rocauc = 0.0 # Assign a default value

    # TODO: Calculate image-level ROC AUC using det_scores
    # Need to review later
    image_level_rocauc = 0.0
    image_scores = [score.max() for score in det_scores]
    if len(np.unique(gt_list)) > 1:
        image_level_rocauc = roc_auc_score(gt_list, image_scores)
        print(f"Image-level ROC AUC: {image_level_rocauc:.4f}")
    else:
        print("Warning: Could not calculate image ROC AUC. Need both normal and anomalous samples.")


    # --- Save Debug Images ---
    if debug_image_indices:
        print(f"Saving debug images to {save_dir}...")
        save_debug_images(
            indices=debug_image_indices,
            test_imgs=test_imgs_list,
            recon_imgs=recon_imgs_list,
            seg_scores=seg_scores_normalized,
            gt_masks=gt_mask_np,
            mean=mean_train,
            std=std_train,
            save_dir=save_dir,
            class_name=class_name,
            model_name=args.model
        )

    return per_pixel_rocauc, image_level_rocauc

def save_debug_images(
    indices: List[int],
    test_imgs: List[np.ndarray],
    recon_imgs: List[np.ndarray],
    seg_scores: np.ndarray, # Normalized scores (N, H, W)
    gt_masks: np.ndarray,   # Ground truth masks (N, H, W)
    mean: List[float],
    std: List[float],
    save_dir: Path,
    class_name: str,
    model_name: str
):
    """Saves comparison images for debugging."""
    num_images = len(test_imgs)
    save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    for idx in indices:
        if idx < 0 or idx >= num_images:
            print(f"Warning: Debug image index {idx} out of bounds (0-{num_images-1}). Skipping.")
            continue

        img_orig = test_imgs[idx]
        img_recon = recon_imgs[idx]
        score_map = seg_scores[idx]
        gt_mask = gt_masks[idx]

        # Denormalize images for visualization (assuming CHW format from ToTensor)
        # Ensure denormalization handles numpy arrays correctly
        img_orig_vis = denormalization(img_orig, mean, std)
        img_recon_vis = denormalization(img_recon, mean, std)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4)) # Adjusted figsize
        fig.suptitle(f"Debug Image {idx} - Class: {class_name}")

        axes[0].imshow(img_orig_vis)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(img_recon_vis)
        axes[1].set_title("Reconstructed Image")
        axes[1].axis('off')

        im = axes[2].imshow(score_map, cmap='jet', vmin=0, vmax=1) # Use normalized score map
        axes[2].set_title("Anomaly Map")
        axes[2].axis('off')

        axes[3].imshow(gt_mask, cmap='gray')
        axes[3].set_title("Ground Truth Mask")
        axes[3].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        save_path = save_dir / f"debug_{model_name}_{class_name}_{idx}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved debug image: {save_path}")


def run_experiment(
    args: Any,
    class_name: str,
    base_save_dir: Path,
    device: torch.device,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
) -> Dict[str, Any]:
    """Runs the full training and evaluation pipeline for a single class."""

    print("\n" + "="*50)
    print(f" Staring Experiment for Class: {class_name} ")
    print("="*50 + "\n")

    start_time = time.time()

    # --- Setup for the specific class ---
    args.obj = class_name # Set the class name in args for dataloader
    set_seed(args.seed) # Set seed for reproducibility for this specific run

    # Create specific save directory for this class run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_save_dir = base_save_dir / f"{class_name}_{args.model}_{run_timestamp}"
    current_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {current_save_dir}")

    # --- Data Loaders ---
    print("Loading datasets...")
    try:
        train_loader, val_loader, test_loader = get_dataloader(args)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Error getting dataloaders for class {class_name}: {e}")
        return {"class": class_name, "status": "failed", "error": str(e)}

    # --- Model, Optimizer, Criterion, Scaler ---
    model = get_model(device)
    optimizer = get_optimizer(model)
    criterion = nn.MSELoss()
    #criterion = L2Loss()
    #criterion = CosLoss()

    scaler = amp.GradScaler(enabled=args.amp)

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(args, model, train_loader, optimizer, criterion, scaler, device, epoch)

        if epoch % args.val_epochs == 0:
            val_loss = validate_epoch(args, model, val_loader, criterion, device, epoch)
            # Log metrics to WandB (if enabled)
            if wandb_run:
                wandb.log({
                    f"{class_name}_train_loss": train_loss,
                    f"{class_name}_val_loss": val_loss,
                    "epoch": epoch  # Log global epoch step if needed, or per-class epoch
                })

            # Simple best model saving based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the best model checkpoint
                best_model_path = current_save_dir / f"model_{class_name}_{args.model}_best.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"Validation loss improved to {val_loss:.6f}. Saved best model to {best_model_path}")
            else:
                epochs_without_improvement += 1

            if 0 < args.early_stopping_patience <= epochs_without_improvement:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        else:
            if wandb_run:
                wandb.log({
                    f"{class_name}_train_loss": train_loss,
                    "epoch": epoch  # Log global epoch step if needed, or per-class epoch
                })

    print("Training finished.")

    # --- Final Model Saving ---
    final_model_path = current_save_dir / f"model_{class_name}_{args.model}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    # --- Evaluation ---
    # Load the best model for evaluation
    print(f"Loading best model from {best_model_path} for evaluation...")
    # Re-instantiate model structure and load state dict
    eval_model = get_model(device) # Get a fresh instance
    try:
        eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
    except FileNotFoundError:
        print(f"Warning: Best model checkpoint not found at {best_model_path}. Using final model for evaluation.")
        eval_model.load_state_dict(torch.load(final_model_path, map_location=device)) # Fallback to final model

    pixel_auc, image_auc = evaluate_performance(
        model=eval_model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        save_dir=current_save_dir,
        class_name=class_name
    )

    end_time = time.time()
    duration = end_time - start_time
    print(f"Experiment for class {class_name} finished in {duration:.2f} seconds.")

    # --- Results ---
    results = {
        "class": class_name,
        "status": "completed",
        "pixel_rocauc": pixel_auc,
        "image_rocauc": image_auc,
        "training_time_seconds": duration,
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
        "save_dir": str(current_save_dir)
    }

    # Log final metrics for this class to WandB Summary
    if wandb_run:
        wandb.summary[f"{class_name}_pixel_rocauc"] = pixel_auc
        wandb.summary[f"{class_name}_image_rocauc"] = image_auc
        wandb.summary[f"{class_name}_training_time_seconds"] = duration

    return results


if __name__ == "__main__":

    # --- Global Setup ---
    start_time_full_training = time.time()
    base_save_dir = Path(args.save_dir)
    base_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base save directory: {base_save_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}") # Assumes single GPU or GPU 0

    # --- Wandb Initialization ---
    wandb_run = None
    if args.wandb_log:
        try:
            run_name = f"{args.model}_{args.obj if args.obj != 'all' else 'full'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(
                project=WANDB_PROJECT_NAME,
                name=run_name,
                notes=args.log_comment,
                tags=WANDB_TAGS + [args.model, args.obj],
                config=vars(args), # Log all arguments from config
                dir=str(base_save_dir), # Set wandb log directory
                mode=args.mode if args.mode is not None else "disabled",
                settings=wandb.Settings(start_method="fork")
            )
            print(f"Wandb initialized. Run name: {run_name}")
        except Exception as e:
            print(f"Error initializing Wandb: {e}. Wandb logging disabled.")
            wandb_run = None # Ensure it's None if init fails

    # --- Determine Classes to Run ---
    if args.obj == 'all': # Run all class separately
        classes_to_run = CLASS_NAMES
        print(f"Running experiments for all classes: {classes_to_run}")
    elif args.obj in CLASS_NAMES:
        classes_to_run = [args.obj]
        print(f"Running experiment for single class: {args.obj}")
    else:
        print(f"Error: Object '{args.obj}' not found in CLASS_NAMES: {CLASS_NAMES}")
        sys.exit(1)

    # --- Run Experiments ---
    all_results = []
    try:
        for class_name in classes_to_run:
            class_results = run_experiment(
                args=args,
                class_name=class_name,
                base_save_dir=base_save_dir,
                device=device,
                wandb_run=wandb_run
            )
            all_results.append(class_results)

        # --- Final Summary ---
        print("\n" + "="*50)
        print(" Overall Results Summary ")
        print("="*50)
        print(f"{'Class':<15} | {'Status':<10} | {'Pixel ROCAUC':<15} | {'Image ROCAUC':<15} | {'Time (s)':<10}")
        print("-"*70)

        total_pixel_auc = 0.0
        total_image_auc = 0.0
        completed_count = 0

        for result in all_results:
            if result['status'] == 'completed':
                print(f"{result['class']:<15} | {result['status']:<10} | {result['pixel_rocauc']:.4f}{'':<9} | {result['image_rocauc']:.4f}{'':<9} | {result['training_time_seconds']:.2f}")
                total_pixel_auc += result['pixel_rocauc']
                total_image_auc += result['image_rocauc']
                completed_count += 1
            else:
                print(f"{result['class']:<15} | {result['status']:<10} | {'N/A':<15} | {'N/A':<15} | {'N/A':<10}")
                print(f"  Error: {result.get('error', 'Unknown')}")

        if completed_count > 0:
            avg_pixel_auc = total_pixel_auc / completed_count
            avg_image_auc = total_image_auc / completed_count
            print("-"*70)
            print(f"{'Average':<15} | {'':<10} | {avg_pixel_auc:.4f}{'':<9} | {avg_image_auc:.4f}{'':<9} |")
            print("="*70)
            if wandb_run:
                wandb.summary["average_pixel_rocauc"] = avg_pixel_auc
                wandb.summary["average_image_rocauc"] = avg_image_auc
                wandb.summary["completed_classes"] = completed_count
        else:
            print("\nNo experiments completed successfully.")

        total_duration = time.time() - start_time_full_training
        print(f"\nTotal script execution time: {total_duration:.2f} seconds.")
        if wandb_run:
            wandb.summary["total_duration_seconds"] = total_duration

    finally:
        # --- Finish Wandb Run ---
        if wandb_run:
            wandb.finish()
            print("Wandb run finished.")
