# -*- coding: utf-8 -*-
"""
MVTec AD Anomaly Detection Training and Evaluation Pipeline
Using Vision Transformer (ViT) Encoder-Decoder Variants.

TIMESTAMP @ 2025-04-10T23:45:47
author: phuocddat (Refactored with metric.py integration)
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
import logging # Use standard logging

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from typing import Tuple, List, Dict, Any, Optional, Union

# --- Local Imports ---
try:
    from dataset_mvtec import get_dataloader
    from model_vitad import load_default_model # Using ViTAD model with manual loading function
    from losses import L2Loss # L1Loss, CosLoss also available
    from utils_mvtec import set_seed, denormalization, log_write # Keep utility functions
    from config import args, CLASS_NAMES, mean_train, std_train
    # Import the refactored Evaluator
    from metrics import Evaluator
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# --- Configuration ---
ANOMALY_MAP_SIGMA = 4
# Metrics to compute using the Evaluator
# Choose a subset relevant to your goals, or use Evaluator's default
METRICS_TO_COMPUTE = [
    'mAUROC_px', 'mAUROC_sp_max', # Pixel and Image AUROC (max pooling)
    'mAUPRO_px',                 # Pixel AUPRO
    'mAP_px', 'mAP_sp_max',      # Pixel and Image Average Precision
    'mF1_max_sp_max',            # Max F1 for Image (max pooling)
    'mIoU_max_px',               # Max Pixel IoU over thresholds
]
WANDB_PROJECT_NAME = "INM705-exp"
WANDB_TAGS = ["experiment", "ViT-AD", "ViTAD custom model"]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# %% Helper Functions

def get_model(device: torch.device) -> nn.Module:
    """Instantiates and returns the anomaly detection model."""
    model = load_default_model().to(device)
    logger.info(f"Using model: {model.__class__.__name__}")
    try:
        # Debug model summary
        summary(model, input_size=(args.batch_size, 3, args.image_size, args.image_size))
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")
    return model

def get_optimizer(model: nn.Module) -> optim.Optimizer:
    """Instantiates and returns the optimizer."""
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )
    logger.info(f"Using optimizer: {optimizer.__class__.__name__} with LR: {args.lr}")
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
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)

    for x, _, _ in pbar:
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True) # More efficient zeroing

        if args.amp:
            with amp.autocast():
                feature_enc, feature_fus = model(x) # Get features from ViTAD model
                loss = criterion(feature_enc, feature_fus) # Compare features
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            feature_enc, feature_fus = model(x)
            loss = criterion(feature_enc, feature_fus)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    # logger.info(f"Epoch {epoch}/{args.epochs} [Train] Avg Loss: {avg_loss:.6f}")
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
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False)

    with torch.no_grad():
        for x, _, _ in pbar:
            x = x.to(device)
            if args.amp:
                with amp.autocast():
                    feature_enc, feature_fus = model(x)
                    loss = criterion(feature_enc, feature_fus)
            else:
                feature_enc, feature_fus = model(x)
                loss = criterion(feature_enc, feature_fus)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    # logger.info(f"Epoch {epoch}/{args.epochs} [Val] Avg Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate_performance(
    model: nn.Module,
    test_loader: DataLoader,
    epoch_number: Union[int, None],
    device: torch.device,
    save_dir: Path,
    class_name: str,
    evaluator: Evaluator, # Pass the evaluator instance
    debug_image_indices: Optional[List[int]] = [5, 10, 15] # Indices to save debug images
) -> Dict[str, float]:
    """
    Runs the full test evaluation pipeline using the provided Evaluator.

    Returns:
        Dict[str, float]: Dictionary of calculated metrics.
    """
    model.eval()

    # --- Data Collection for Evaluator ---
    all_gt_masks = []       # List to store ground truth pixel masks (np arrays)
    all_anomaly_maps = []   # List to store predicted anomaly maps (np arrays)
    all_gt_labels = []      # List to store ground truth image labels (0 or 1)
    all_class_names = []    # List to store class names (should be uniform for test_loader)
    test_imgs_for_debug = [] # Store original images only if needed for debugging
    if epoch_number is None:
        epoch_number = 'final stage'
    logger.info(f"Running evaluation for class: {class_name}... epoch: {str(epoch_number)}")
    pbar = tqdm(test_loader, desc=f"Evaluation [{class_name}]", leave=False)

    output_size = (args.image_size, args.image_size)

    for (x, label, mask) in pbar:
        if debug_image_indices:
             test_imgs_for_debug.extend(x.cpu().numpy()) # Store original images

        x = x.to(device)
        with torch.no_grad():
            if args.amp:
                 with amp.autocast():
                    feature_enc, feature_fus = model(x) # Get features
            else:
                feature_enc, feature_fus = model(x) # Get features

        # --- Anomaly Map Calculation using Evaluator's static method ---
        # Pass features (ensure they are lists), desired output size, sigma etc.
        # Dim: [Batch, Channels, H, W]
        if not isinstance(feature_enc, list): feature_enc = [feature_enc]
        if not isinstance(feature_fus, list): feature_fus = [feature_fus]

        # # Check if features exist before calculating map
        # Uncomment for debugging purpose
        # ## TIMESTAMP @ 2025-04-19T08:47:13
        ## author: phuocddat
        ## start
        # if not feature_enc or not feature_fus:
        #      logger.error("Model did not return valid features for anomaly map calculation. Skipping batch.")
        #      continue
        ## end --


        anomaly_map_batch, _ = Evaluator.cal_anomaly_map(
            ft_list=feature_enc,
            fs_list=feature_fus,
            out_size=output_size,
            uni_am=False, # Or True, depending on desired fusion strategy
            amap_mode='add', # Or 'mul'
            gaussian_sigma=ANOMALY_MAP_SIGMA, # Use configured sigma
            use_cos=False # Or False for L2 distance - match training loss if relevant
        )
        # anomaly_map_batch shape should be (B, H, W)

        # --- Store results for the Evaluator ---
        all_anomaly_maps.extend(anomaly_map_batch) # Add batch of maps (numpy arrays)
        all_gt_masks.extend(mask.squeeze(1).cpu().numpy()) # Remove channel dim, move to CPU (N, H, W)
        all_gt_labels.extend(label.cpu().numpy()) # (N,)
        all_class_names.extend([class_name] * x.size(0)) # Add class name for each sample in batch

    # --- Prepare data for Evaluator ---
    if not all_anomaly_maps:
        logger.error(f"No anomaly maps generated for class {class_name}. Cannot evaluate.")
        return {}

    results_dict = {
        'imgs_masks': np.stack(all_gt_masks, axis=0).astype(np.uint8), # Stack to (N, H, W)
        'anomaly_maps': np.stack(all_anomaly_maps, axis=0),     # Stack to (N, H, W)
        'anomalys': np.stack(all_gt_labels, axis=0),             # Stack to (N,)
        'cls_names': np.array(all_class_names)
        # disable for now
        # Add 'smp_pre', 'smp_masks' if your model provides direct sample-level predictions
    }

    # --- Run the Evaluator ---
    metrics = evaluator.run(results_dict, class_name)

    # --- Save Debug Images ---
    if debug_image_indices and test_imgs_for_debug:
        logger.info(f"Saving debug images to {save_dir}...")
        # Normalize anomaly maps for consistent visualization if needed
        # The evaluator uses raw maps, but for viz, normalization might be better
        viz_anomaly_maps = results_dict['anomaly_maps']
        map_min, map_max = viz_anomaly_maps.min(), viz_anomaly_maps.max()
        if map_max > map_min:
             viz_anomaly_maps_norm = (viz_anomaly_maps - map_min) / (map_max - map_min)
        else:
             viz_anomaly_maps_norm = np.zeros_like(viz_anomaly_maps)

        save_debug_images(
            indices=debug_image_indices,
            test_imgs=test_imgs_for_debug, # Original images
            epoch_number=epoch_number,
            #TODO: Add reconstruct func from intermediate features
            # recon_imgs=None, # Removed as ViTAD doesn't reconstruct by default
            anomaly_maps=viz_anomaly_maps_norm, # Use normalized maps for viz
            gt_masks=results_dict['imgs_masks'], # Ground truth masks
            mean=mean_train, # Denormalization params
            std=std_train,
            save_dir=save_dir,
            class_name=class_name,
            model_name=args.model
        )

    # Log the metrics calculated by the evaluator
    logger.info(f"Evaluation Metrics for {class_name}: {metrics}")

    return metrics


def save_debug_images(
    indices: List[int],
    test_imgs: List[np.ndarray], # List of (C, H, W) numpy arrays
    epoch_number: Union[int, None],
    anomaly_maps: np.ndarray, # All anomaly maps (N, H, W) - expected normalized [0,1] for viz
    gt_masks: np.ndarray,   # All ground truth masks (N, H, W)
    mean: List[float],
    std: List[float],
    save_dir: Path,
    class_name: str,
    model_name: str
):
    """Saves comparison images for debugging."""
    num_images = len(test_imgs)
    save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    plot_cols = 3 # Original, Anomaly Map, GT Mask
    fig_width = 12
    if epoch_number is None:
        epoch_number = 'final'
    for idx in indices:
        if idx < 0 or idx >= num_images:
            logger.warning(f"Debug image index {idx} out of bounds (0-{num_images-1}). Skipping.")
            continue

        img_orig_chw = test_imgs[idx]
        score_map = anomaly_maps[idx]
        gt_mask = gt_masks[idx]

        # Denormalize original image (CHW -> HWC for display)
        img_orig_vis = denormalization(img_orig_chw, mean, std) # Assumes denormalization returns HWC

        fig, axes = plt.subplots(1, plot_cols, figsize=(fig_width, 4)) # Adjusted figsize
        fig.suptitle(f"Debug Image {idx} - Class: {class_name}")

        axes[0].imshow(img_orig_vis)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Display Anomaly Map
        im = axes[1].imshow(score_map, cmap='jet', vmin=0, vmax=1) # Assume normalized map
        axes[1].set_title("Anomaly Map")
        axes[1].axis('off')

        axes[2].imshow(gt_mask, cmap='gray')
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        save_path = save_dir / f"debug_{model_name}_{class_name}_{idx}_{str(epoch_number)}.png"
        try:
            plt.savefig(save_path)
        except Exception as e:
            logger.error(f"Failed to save debug image {save_path}: {e}")
        plt.close(fig)


def run_experiment(
    args: Any,
    class_name: str,
    base_save_dir: Path,
    device: torch.device,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
) -> Dict[str, Any]:
    """Runs the full training and evaluation pipeline for a single class."""

    logger.info("\n" + "="*50)
    logger.info(f" Starting Experiment for Class: {class_name} ")
    logger.info("="*50 + "\n")

    start_time = time.time()

    # --- Setup for the specific class ---
    args.obj = class_name # Set the class name in args for dataloader
    set_seed(args.seed) # Set seed for reproducibility for this specific run

    # Create specific save directory for this class run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_save_dir = base_save_dir / f"{class_name}_{args.model}_{run_timestamp}"
    current_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to: {current_save_dir}")

    # --- Data Loaders ---
    logger.info("Loading datasets...")
    try:
        train_loader, val_loader, test_loader = get_dataloader(args)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        logger.error(f"Error getting dataloaders for class {class_name}: {e}")
        return {"class": class_name, "status": "failed", "error": str(e)}

    # --- Model, Optimizer, Criterion, Scaler ---
    model = get_model(device)
    optimizer = get_optimizer(model)
    # Using L2Loss between features as per ViTAD approach
    criterion = L2Loss()
    scaler = amp.GradScaler(enabled=args.amp)

    # --- Instantiate Evaluator ---
    evaluator = Evaluator(
        metrics=METRICS_TO_COMPUTE,
        pooling_ks=None, # Adjust if needed
        max_step_aupro=100, # Default
        mp=False, # Set to True to try multiprocessing for AUPRO (can be slow)
        use_adeval=args.use_adeval if hasattr(args, 'use_adeval') else False # Check if arg exists
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = current_save_dir / f"model_{class_name}_{args.model}_best.pt"

    logger.info(f"Starting training for {args.epochs} epochs...")
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
            # debug images here
            eval_metrics = evaluate_performance(
                model=model,
                test_loader=test_loader,
                epoch_number=epoch,
                device=device,
                save_dir=current_save_dir / "eval_debug",  # Subdir for debug images
                class_name=class_name,
                evaluator=evaluator  # Pass the evaluator instance
            )

            if 0 < args.early_stopping_patience <= epochs_without_improvement:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        else:
            if wandb_run:
                wandb.log({
                    f"{class_name}_train_loss": train_loss,
                    "epoch": epoch  # Log global epoch step if needed, or per-class epoch
                })

        # val_loss = validate_epoch(args, model, val_loader, criterion, device, epoch) # Validate every epoch
        #
        # log_dict = {
        #     f"{class_name}/train_loss": train_loss,
        #     f"{class_name}/val_loss": val_loss,
        #     "epoch": epoch
        # }
        # # Simple print for epoch summary
        # print(f"Epoch {epoch}/{args.epochs} => Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        #
        # # Log metrics to WandB (if enabled)
        # if wandb_run:
        #     wandb.log(log_dict)
        #
        # # Simple best model saving based on validation loss
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_without_improvement = 0
        #     # Save the best model checkpoint
        #     torch.save(model.state_dict(), best_model_path)
        #     logger.info(f"Validation loss improved to {val_loss:.6f}. Saved best model to {best_model_path}")
        # else:
        #     epochs_without_improvement += 1

        # if epoch % args.val_epochs == 0:

        # if 0 < args.early_stopping_patience <= epochs_without_improvement:
        #     logger.info(f"Early stopping triggered after {epoch} epochs.")
        #     break


    logger.info("Training finished.")

    # --- Final Model Saving ---
    final_model_path = current_save_dir / f"model_{class_name}_{args.model}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # --- Evaluation ---
    # Load the best model for evaluation
    if best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path} for evaluation...")
        load_path = best_model_path
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}. Using final model for evaluation.")
        load_path = final_model_path

    # Re-instantiate model structure and load state dict
    eval_model = get_model(device) # Get a fresh instance
    try:
        eval_model.load_state_dict(torch.load(load_path, map_location=device))
    except Exception as e:
        logger.error(f"Failed to load model state dict from {load_path}: {e}")
        return {"class": class_name, "status": "failed", "error": f"Model loading failed: {e}"}

    eval_metrics = evaluate_performance(
        model=eval_model,
        test_loader=test_loader,
        epoch_number= None,
        # criterion=criterion, # Not needed directly if using features
        device=device,
        save_dir=current_save_dir / "eval_debug", # Subdir for debug images
        class_name=class_name,
        evaluator=evaluator # Pass the evaluator instance
    )

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Experiment for class {class_name} finished in {duration:.2f} seconds.")

    # --- Results ---
    results = {
        "class": class_name,
        "status": "completed" if eval_metrics else "evaluation_failed",
        "metrics": eval_metrics, # Store the dict of metrics
        "training_time_seconds": duration,
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path) if best_model_path.exists() else "N/A",
        "save_dir": str(current_save_dir)
    }

    # Log final metrics for this class to WandB Summary
    if wandb_run and eval_metrics:
        for metric_name, metric_value in eval_metrics.items():
            wandb.summary[f"{class_name}/{metric_name}"] = metric_value
        wandb.summary[f"{class_name}/training_time_seconds"] = duration

    return results


if __name__ == "__main__":

    # --- Global Setup ---
    start_time_full_training = time.time()
    args_save_dir = Path(args.save_dir)
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_save_dir = Path(args_save_dir) / f"{run_timestamp}"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Base save directory: {base_save_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    # --- Wandb Initialization ---
    wandb_run = None
    # Check if wandb logging is enabled in args and mode is not 'disabled'
    if args.wandb_log and (args.mode is None or args.mode.lower() != 'disabled'):
        try:
            run_name = f"{args.model}_{args.obj if args.obj != 'all' else 'full'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(
                project=WANDB_PROJECT_NAME,
                name=run_name,
                tags=WANDB_TAGS + [args.model, args.obj],
                config=vars(args), # Log all arguments
                dir=str(base_save_dir), # Set wandb log directory
                settings=wandb.Settings(start_method="fork")
            )
            logger.info(f"Wandb initialized. Run name: {run_name}")
        except Exception as e:
            logger.error(f"Error initializing Wandb: {e}. Wandb logging disabled.")
            wandb_run = None # Ensure it's None if init fails
    else:
        logger.info("Wandb logging is disabled.")


    # --- Determine Classes to Run ---
    if args.obj == 'all': # Run all classes separately
        classes_to_run = CLASS_NAMES
        logger.info(f"Running experiments for all MVTec AD classes: {classes_to_run}")
    elif args.obj in CLASS_NAMES:
        classes_to_run = [args.obj]
        logger.info(f"Running experiment for single class: {args.obj}")
    else:
        logger.error(f"Error: Object '{args.obj}' not found in CLASS_NAMES: {CLASS_NAMES}")
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
        logger.info("\n" + "="*80)
        logger.info(" Overall Results Summary ")
        logger.info("="*80)

        # Define headers based on the metrics actually computed
        header_metrics = METRICS_TO_COMPUTE # Use the list defined earlier
        header = f"{'Class':<15} | {'Status':<10} | " + " | ".join([f'{m:<15}' for m in header_metrics]) + f" | {'Time (s)':<10}"
        logger.info(header)
        logger.info("-" * len(header))

        # Aggregate metrics for average calculation
        aggregated_metrics = {m: [] for m in header_metrics}
        completed_count = 0
        total_time = 0.0

        for result in all_results:
            status = result['status']
            class_name = result['class']
            time_s = result.get('training_time_seconds', 0)

            if status == 'completed':
                metrics = result['metrics']
                metric_values = [metrics.get(m, np.nan) for m in header_metrics] # Get values, use NaN if missing
                row = f"{class_name:<15} | {status:<10} | " + " | ".join([f'{v:.4f}' if not np.isnan(v) else 'N/A' for v in metric_values]) + f" | {time_s:<10.2f}"
                logger.info(row)

                for m, v in zip(header_metrics, metric_values):
                    if not np.isnan(v):
                        aggregated_metrics[m].append(v)
                completed_count += 1
                total_time += time_s
            else:
                error_msg = result.get('error', 'Unknown')
                row = f"{class_name:<15} | {status:<10} | " + " | ".join(['N/A'] * len(header_metrics)) + f" | {'N/A':<10}"
                logger.info(row)
                logger.warning(f"  -> Failed Class '{class_name}' Reason: {error_msg}")


        if completed_count > 0:
             avg_metrics = {m: np.mean(vals) for m, vals in aggregated_metrics.items() if vals}
             avg_metric_values = [avg_metrics.get(m, np.nan) for m in header_metrics]
             avg_time = total_time / completed_count

             logger.info("-" * len(header))
             avg_row = f"{'Average':<15} | {'':<10} | " + " | ".join([f'{v:.4f}' if not np.isnan(v) else 'N/A' for v in avg_metric_values]) + f" | {avg_time:<10.2f}"
             logger.info(avg_row)
             logger.info("=" * len(header))

             # Log average metrics to WandB
             if wandb_run:
                 for m, v in avg_metrics.items():
                     wandb.summary[f"average/{m}"] = v
                 wandb.summary["average/training_time_seconds"] = avg_time
                 wandb.summary["completed_classes"] = completed_count
        else:
             logger.info("\nNo experiments completed successfully.")

        total_duration_script = time.time() - start_time_full_training
        logger.info(f"\nTotal script execution time: {total_duration_script:.2f} seconds.")
        if wandb_run:
            wandb.summary["total_duration_seconds"] = total_duration_script

    except Exception as e:
         logger.error(f"An unexpected error occurred during the main execution: {e}", exc_info=True)
    finally:
        # --- Finish Wandb Run ---
        if wandb_run:
            wandb.finish()
            logger.info("Wandb run finished.")
