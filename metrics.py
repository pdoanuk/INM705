import time
import multiprocessing
import copy
import logging # Use standard logging

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from skimage import measure
import torch
from torch.nn import functional as F

# Optional dependency, only imported if used
try:
    from adeval import EvalAccumulatorCuda
    HAS_ADEVAL = True
except ImportError:
    HAS_ADEVAL = False
    EvalAccumulatorCuda = None # Placeholder

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator(object):
    """
    Calculates various anomaly detection metrics.

    Args:
        metrics (list, optional): List of metrics to calculate.
            Defaults to a comprehensive list.
        pooling_ks (int, optional): Kernel size for average pooling before
            calculating sample-level scores from pixel maps. Defaults to None.
        max_step_aupro (int, optional): Number of steps (thresholds) for AUPRO
            calculation. Defaults to 200.
        mp (bool, optional): Whether to use multiprocessing for AUPRO calculation.
            Defaults to False. Can be slow.
        use_adeval (bool, optional): Whether to use the external adeval library
            for calculation (requires installation and CUDA). Defaults to False.
    """
    def __init__(self, metrics=None, pooling_ks=None, max_step_aupro=200, mp=False, use_adeval=False):
        if metrics is None:
            self.metrics = [
                'mAUROC_sp_max', 'mAUROC_px', 'mAUPRO_px',
                'mAP_sp_max', 'mAP_px',
                'mF1_max_sp_max',
                'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
                'mF1_max_px', 'mIoU_max_px',
            ]
        else:
            self.metrics = metrics

        self.pooling_ks = pooling_ks
        self.max_step_aupro = max_step_aupro
        self.mp = mp # Note: multiprocessing for AUPRO can be slow due to overhead

        self.eps = 1e-8
        self.beta = 1.0 # Beta for F1 score

        self.boundary = 1e-7 # For adeval score range adjustment
        self.use_adeval = use_adeval and HAS_ADEVAL
        if use_adeval and not HAS_ADEVAL:
            logger.warning("adeval library not found, falling back to standard evaluation.")
            self.use_adeval = False
        if use_adeval and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to standard evaluation (adeval requires CUDA).")
            self.use_adeval = False


    def run(self, results, cls_name):
        """
        Calculates metrics for a given class based on provided results.

        Args:
            results (dict): A dictionary containing evaluation data:
                - 'imgs_masks' (np.ndarray): Ground truth pixel masks (N, H, W).
                - 'anomaly_maps' (np.ndarray): Predicted anomaly maps (N, H, W).
                - 'cls_names' (np.ndarray): Array of class names for each sample (N,).
                - 'anomalys' (np.ndarray): Ground truth image-level labels (0 or 1) (N,).
                - 'smp_pre' (np.ndarray, optional): Sample-level predictions (if available, e.g., from classifier head).
                - 'smp_masks' (np.ndarray, optional): Ground truth sample-level labels (if available).
            cls_name (str): The name of the class being evaluated.

        Returns:
            dict: A dictionary mapping metric names to their calculated values.
        """
        start_run_time = time.time()
        # Filter results for the specified class
        logger.info(f"Evaluating {cls_name}...")
        if self.use_adeval:
            logger.info("Evaluation with adeval support")
        else:
            logger.warning("Evaluation without adeval support")

        idxes = results['cls_names'] == cls_name
        if not np.any(idxes):
            logger.warning(f"No samples found for class '{cls_name}'. Skipping evaluation.")
            return {}

        gt_px = results['imgs_masks'][idxes]
        pr_px = results['anomaly_maps'][idxes]
        gt_sp = results['anomalys'][idxes] # Image-level GT labels

        # Handle optional sample-level predictions if provided directly
        pr_sp = None
        if 'smp_pre' in results:
             # Assuming 'smp_pre' might be direct image-level scores
             pr_sp = results['smp_pre'][idxes]

        # Reshape masks/maps if necessary (e.g., remove channel dim)
        if gt_px.ndim == 4 and gt_px.shape[1] == 1:
            gt_px = gt_px.squeeze(1)
        if pr_px.ndim == 4 and pr_px.shape[1] == 1:
            pr_px = pr_px.squeeze(1)

        if gt_px.shape[0] == 0 or pr_px.shape[0] == 0:
             logger.warning(f"Not enough samples for class '{cls_name}' after filtering. Skipping.")
             return {}

        # --- Calculate Image-Level Scores from Pixel Maps ---
        # Normalize pixel maps for consistent thresholding in some metrics
        pr_px_min = pr_px.min()
        pr_px_max = pr_px.max()
        if pr_px_max - pr_px_min > self.eps:
             pr_px_norm = (pr_px - pr_px_min) / (pr_px_max - pr_px_min)
        else:
             logger.warning(f"Anomaly map range is too small for class '{cls_name}'. Normalization might be unstable.")
             pr_px_norm = np.zeros_like(pr_px) # Avoid division by zero

        # Calculate image scores (max and mean) from pixel maps
        if self.pooling_ks is not None and self.pooling_ks > 1:
            # Ensure tensor is on CPU for pooling if needed, expects NCHW
            pr_px_torch = torch.from_numpy(pr_px).unsqueeze(1).float()
            pr_px_pooling = F.avg_pool2d(pr_px_torch, self.pooling_ks, stride=1, padding=self.pooling_ks//2).squeeze(1).numpy()
            pr_sp_max = pr_px_pooling.max(axis=(1, 2))
            pr_sp_mean = pr_px_pooling.mean(axis=(1, 2))
        else:
            pr_sp_max = pr_px.max(axis=(1, 2))
            pr_sp_mean = pr_px.mean(axis=(1, 2))

        # Use pre-calculated sample scores if available and no specific map-derived score requested
        # This logic might need adjustment based on how 'smp_pre' is intended to be used
        if pr_sp is None:
            pr_sp = pr_sp_max # Default to max pixel value if no other sample score is given

        # --- Use adeval if enabled ---
        adeval_metrics = {}
        if self.use_adeval:
            try:
                logger.info(f"Using adeval for {cls_name}...")
                score_min = min(pr_sp_max) - self.boundary
                score_max = max(pr_sp_max) + self.boundary
                anomap_min = pr_px.min()
                anomap_max = pr_px.max()
                # Note: adeval might have specific requirements for nstrips/other params
                accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=50)
                # Process in batches for potentially large datasets
                batch_size = 32 # Adjust as needed
                for i in range(0, pr_px.shape[0], batch_size):
                     batch_pr_px = torch.from_numpy(pr_px[i:i+batch_size]).cuda(non_blocking=True)
                     batch_gt_px = torch.from_numpy(gt_px[i:i+batch_size].astype(np.uint8)).cuda(non_blocking=True)
                     accum.add_anomap_batch(batch_pr_px, batch_gt_px)

                for i in range(len(pr_sp_max)):
                    # adeval expects individual image scores and labels
                    accum.add_image(torch.tensor(pr_sp_max[i]), torch.tensor(gt_sp[i].item())) # Ensure label is scalar

                adeval_metrics = accum.summary()
                logger.info(f"adeval summary for {cls_name}: {adeval_metrics}")
            except Exception as e:
                logger.error(f"Error during adeval calculation for {cls_name}: {e}. Falling back to standard metrics.")
                self.use_adeval = False # Disable for subsequent metrics if it failed once

        # --- Calculate Metrics ---
        metric_results = {}
        metric_times = {}
        for metric in self.metrics:
            t0 = time.time()
            try:
                if metric == 'mAUROC_sp_max':
                    if self.use_adeval and 'i_auroc' in adeval_metrics:
                        metric_results[metric] = adeval_metrics['i_auroc']
                    elif len(np.unique(gt_sp)) > 1:
                        metric_results[metric] = roc_auc_score(gt_sp, pr_sp_max)
                    else:
                         metric_results[metric] = 0.0
                         logger.warning(f"Skipping {metric} for {cls_name}: Only one class present in image labels.")
                elif metric == 'mAUROC_sp_mean':
                    if len(np.unique(gt_sp)) > 1:
                        metric_results[metric] = roc_auc_score(gt_sp, pr_sp_mean)
                    else:
                         metric_results[metric] = 0.0
                         logger.warning(f"Skipping {metric} for {cls_name}: Only one class present in image labels.")
                elif metric == 'mAUROC_sp': # Uses the potentially provided sample scores
                     if pr_sp is not None and len(np.unique(gt_sp)) > 1:
                          metric_results[metric] = roc_auc_score(gt_sp, pr_sp)
                     else:
                          metric_results[metric] = 0.0
                          logger.warning(f"Skipping {metric} for {cls_name}: Missing sample predictions or only one class.")

                elif metric == 'mAUROC_px':
                    if self.use_adeval and 'p_auroc' in adeval_metrics:
                        metric_results[metric] = adeval_metrics['p_auroc']
                    else:
                         # Ensure mask has variance
                         if len(np.unique(gt_px)) > 1:
                              metric_results[metric] = roc_auc_score(gt_px.ravel(), pr_px.ravel())
                         else:
                              metric_results[metric] = 0.0
                              logger.warning(f"Skipping {metric} for {cls_name}: Ground truth mask is constant.")
                elif metric == 'mAUPRO_px':
                    if self.use_adeval and 'p_aupro' in adeval_metrics:
                        metric_results[metric] = adeval_metrics['p_aupro']
                    else:
                        # Only calculate if there are positive regions in the ground truth
                        if gt_px.max() > 0:
                            metric_results[metric] = self.cal_pro_score(gt_px, pr_px, max_step=self.max_step_aupro, mp=self.mp)
                        else:
                            metric_results[metric] = 0.0
                            logger.warning(f"Skipping {metric} for {cls_name}: No positive regions in ground truth masks.")

                elif metric == 'mAP_sp_max':
                     if self.use_adeval and 'i_aupr' in adeval_metrics:
                         metric_results[metric] = adeval_metrics['i_aupr']
                     elif len(np.unique(gt_sp)) > 1:
                         metric_results[metric] = average_precision_score(gt_sp, pr_sp_max)
                     else:
                         metric_results[metric] = 0.0
                         logger.warning(f"Skipping {metric} for {cls_name}: Only one class present in image labels.")
                elif metric == 'mAP_sp_mean':
                     if len(np.unique(gt_sp)) > 1:
                         metric_results[metric] = average_precision_score(gt_sp, pr_sp_mean)
                     else:
                         metric_results[metric] = 0.0
                         logger.warning(f"Skipping {metric} for {cls_name}: Only one class present in image labels.")
                elif metric == 'mAP_px':
                    if self.use_adeval and 'p_aupr' in adeval_metrics:
                         metric_results[metric] = adeval_metrics['p_aupr']
                    else:
                         if len(np.unique(gt_px)) > 1:
                             metric_results[metric] = average_precision_score(gt_px.ravel(), pr_px.ravel())
                         else:
                              metric_results[metric] = 0.0
                              logger.warning(f"Skipping {metric} for {cls_name}: Ground truth mask is constant.")

                elif metric == 'mF1_max_sp_max':
                    if len(np.unique(gt_sp)) > 1:
                        precisions, recalls, _ = precision_recall_curve(gt_sp, pr_sp_max)
                        f1_scores = (2 * precisions * recalls) / (precisions + recalls + self.eps)
                        metric_results[metric] = np.max(f1_scores[np.isfinite(f1_scores)]) if len(f1_scores)>0 else 0.0
                    else:
                         metric_results[metric] = 0.0
                         logger.warning(f"Skipping {metric} for {cls_name}: Only one class present in image labels.")
                # Add mF1_max_sa_max if needed
                elif metric.startswith('mF1_px') or metric.startswith('mDice_px') or metric.startswith('mAcc_px') or metric.startswith('mIoU_px'):
                    # Example: mF1_px_0.2_0.8_0.1
                    parts = metric.split('_')
                    use_max = 'max' in parts[1] # Check if it's 'mF1_max_px' etc.

                    if use_max: # e.g., mF1_max_px
                        score_l, score_h, score_step = 0.0, 1.0, 0.05 # Default range for max
                    elif len(parts) == 5: # e.g., mF1_px_0.2_0.8_0.1
                        try:
                             score_l, score_h, score_step = float(parts[-3]), float(parts[-2]), float(parts[-1])
                        except ValueError:
                             logger.error(f"Invalid threshold format for metric: {metric}. Skipping.")
                             continue
                    else:
                        logger.error(f"Invalid format for metric: {metric}. Skipping.")
                        continue

                    gt = gt_px.astype(bool)
                    metric_scores = []
                    # Use normalized anomaly map for thresholding
                    for score in np.arange(score_l, score_h + score_step/2, score_step):
                        pr = pr_px_norm > score
                        # Calculate per-threshold stats (sum over N, H, W)
                        total_area_intersect = np.logical_and(gt, pr).sum()
                        total_area_pred_label = pr.sum()
                        total_area_label = gt.sum()

                        if metric.startswith('mF1_px') or metric.startswith('mDice_px'): # F1 == Dice
                            precision = total_area_intersect / (total_area_pred_label + self.eps)
                            recall = total_area_intersect / (total_area_label + self.eps)
                            f1_px = (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall + self.eps)
                            metric_scores.append(f1_px)
                        elif metric.startswith('mAcc_px'): # Recall
                            recall = total_area_intersect / (total_area_label + self.eps)
                            metric_scores.append(recall)
                        elif metric.startswith('mIoU_px'):
                            total_area_union = np.logical_or(gt, pr).sum()
                            iou_px = total_area_intersect / (total_area_union + self.eps)
                            metric_scores.append(iou_px)
                        else:
                             # Should not happen based on initial check, but as safeguard
                             logger.warning(f"Unhandled threshold metric type: {metric}")

                    metric_scores = np.array(metric_scores)
                    if len(metric_scores) > 0:
                         metric_results[metric] = metric_scores.max() if use_max else metric_scores.mean()
                    else:
                         metric_results[metric] = 0.0
                         logger.warning(f"No scores calculated for threshold metric: {metric}")

            except Exception as e:
                logger.error(f"Error calculating metric '{metric}' for class '{cls_name}': {e}", exc_info=True)
                metric_results[metric] = np.nan # Indicate failure

            t1 = time.time()
            metric_times[metric] = t1 - t0

        # --- Logging ---
        time_str = ', '.join([f"{k}: {v:.3f}s" for k, v in metric_times.items()])
        logger.info(f"Metric Times for {cls_name:<15}: {time_str}")
        logger.info(f"Results for {cls_name:<15}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metric_results.items()]))
        total_time = time.time() - start_run_time
        logger.info(f"Total evaluation time for {cls_name}: {total_time:.3f}s")

        return metric_results

    @staticmethod
    def calculate_image_anomaly_map(x_list, x_hat_list, out_size,
                                    metric='mse',
                                    aggregate_channels='mean',
                                    combination_mode='add',
                                    combination_weights=None,
                                    gaussian_sigma=0,
                                    device: str = 'cpu'
                                    ):
        """
        Calculates pixel-wise anomaly maps between lists of original and reconstructed
        image tensors, combines them, and optionally applies smoothing.

        Args:
            x_list (List[torch.Tensor]): List of original image tensors [B, C, H_i, W_i] (on device).
            x_hat_list (List[torch.Tensor]): List of reconstructed image tensors [B, C, H_i, W_i] (on device).
            out_size (tuple): Target output size for the final map (H_out, W_out).
            metric (str): The metric for pixel difference ('mse' or 'mae'). Defaults to 'mse'.
            aggregate_channels (str): How to aggregate differences across channels ('mean', 'sum', 'max'). Defaults to 'mean'.
            combination_mode (str): How to combine maps from different list items ('add' or 'mul'). Defaults to 'add'.
            combination_weights (Optional[List[float]]): Weights for combining maps if mode is 'add'. Defaults to equal weights.
            gaussian_sigma (float): Sigma for Gaussian smoothing applied to the final combined map. Defaults to 0.
            device (str or torch.device): The device ('cuda' or 'cpu') for calculations.

        Returns:
            torch.Tensor: The final combined anomaly map tensor on the specified device [B, H_out, W_out].
        """
        if not x_list or not x_hat_list or len(x_list) != len(x_hat_list):
            raise ValueError("Input lists x_list and x_hat_list must be non-empty and have the same length.")

        num_maps = len(x_list)
        batch_size = x_list[0].shape[0]

        # Prepare combination weights
        if combination_weights is None or len(combination_weights) != num_maps:
            if combination_weights is not None:
                logger.warning(
                    f"combination_weights length mismatch ({len(combination_weights)} != {num_maps}). Using equal weights.")
            weights = torch.ones(num_maps, device=device, dtype=torch.float32)
        else:
            weights = torch.tensor(combination_weights, device=device, dtype=torch.float32)

        total_weight = weights.sum() if combination_mode == 'add' else 1.0
        if combination_mode == 'add' and total_weight <= 1e-6:
            logger.warning("Sum of combination weights is near zero. Resulting map might be invalid.")
            total_weight = 1.0  # Avoid division by zero

        # Initialize final anomaly map
        if combination_mode == 'add':
            anomaly_map_final = torch.zeros([batch_size, 1, *out_size], dtype=torch.float32, device=device)
        elif combination_mode == 'mul':
            anomaly_map_final = torch.ones([batch_size, 1, *out_size], dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Invalid combination_mode: {combination_mode}. Choose 'add' or 'mul'.")

        individual_anomaly_maps = []  # Optional: store individual maps before combination

        with torch.no_grad():
            for i in range(num_maps):
                x_i = x_list[i].to(device)
                x_hat_i = x_hat_list[i].to(device)

                if x_i.shape != x_hat_i.shape:
                    raise ValueError(
                        f"Shape mismatch between x_list[{i}] ({x_i.shape}) and x_hat_list[{i}] ({x_hat_i.shape}).")
                if x_i.ndim != 4:
                    raise ValueError(f"Input tensors must be 4D (B, C, H, W). Got {x_i.ndim}D for item {i}.")

                # --- Calculate Pixel-wise Difference ---
                if metric == 'mse':
                    diff = (x_i - x_hat_i) ** 2
                elif metric == 'mae':
                    diff = torch.abs(x_i - x_hat_i)
                else:
                    raise ValueError(f"Unsupported metric: {metric}. Choose 'mse' or 'mae'.")

                # --- Aggregate Channels ---
                if aggregate_channels == 'mean':
                    anomaly_map_i = torch.mean(diff, dim=1, keepdim=True)  # [B, 1, H_i, W_i]
                elif aggregate_channels == 'sum':
                    anomaly_map_i = torch.sum(diff, dim=1, keepdim=True)  # [B, 1, H_i, W_i]
                elif aggregate_channels == 'max':
                    anomaly_map_i = torch.max(diff, dim=1, keepdim=True)[0]  # [B, 1, H_i, W_i]
                else:
                    raise ValueError(
                        f"Unsupported channel aggregation: {aggregate_channels}. Choose 'mean', 'sum', or 'max'.")

                # --- Resize individual map to Output Size ---
                anomaly_map_i_resized = F.interpolate(anomaly_map_i, size=out_size, mode='bilinear',
                                                      align_corners=False)  # [B, 1, H_out, W_out]
                individual_anomaly_maps.append(anomaly_map_i_resized)

                # --- Combine with final map ---
                if combination_mode == 'add':
                    anomaly_map_final += anomaly_map_i_resized * weights[i]
                elif combination_mode == 'mul':
                    anomaly_map_final *= torch.clamp(anomaly_map_i_resized, min=1e-6)  # Avoid multiplying by zero

            # Normalize if using 'add' mode
            if combination_mode == 'add':
                anomaly_map_final /= total_weight

            # --- Apply Gaussian Smoothing to the final combined map ---
            if gaussian_sigma > 0:
                logger.debug(f"Using CPU Gaussian filter (sigma={gaussian_sigma}) for combined image anomaly map.")
                anomaly_map_np = anomaly_map_final.squeeze(1).cpu().numpy()
                for idx in range(anomaly_map_np.shape[0]):
                    anomaly_map_np[idx] = gaussian_filter(anomaly_map_np[idx], sigma=gaussian_sigma)
                anomaly_map_final = torch.from_numpy(anomaly_map_np).unsqueeze(1).to(device)

            # Remove channel dimension for final output
            anomaly_map_final = anomaly_map_final.squeeze(1)  # [B, H_out, W_out]


        return anomaly_map_final


    @staticmethod
    def cal_anomaly_map(ft_list, fs_list, out_size, uni_am=False, use_cos=True, amap_mode='add', gaussian_sigma=0, weights=None):
        """
        Calculates anomaly maps by comparing feature lists.

        Args:
            ft_list (list[torch.Tensor]): List of target features.
            fs_list (list[torch.Tensor]): List of source features.
            out_size (list[int] or tuple[int]): Target output size (H, W).
            uni_am (bool): If True, concatenate features before comparison.
            use_cos (bool): If True, use cosine similarity; otherwise, use L2 distance.
            amap_mode (str): 'add' or 'mul' for combining maps from different feature levels.
            gaussian_sigma (float): Sigma for Gaussian smoothing applied to the final map.
            weights (list[float], optional): Weights for combining maps if amap_mode is 'add'.

        Returns:
            tuple:
                - np.ndarray: The calculated anomaly map (N, H, W).
                - list[np.ndarray]: List of individual anomaly maps for each feature level.
        """
        if not ft_list or not fs_list or len(ft_list) != len(fs_list):
            raise ValueError("Feature lists are invalid or have mismatched lengths.")

        # Ensure features are on CPU for numpy conversion later
        ft_list = [f.cpu().detach() for f in ft_list]
        fs_list = [f.cpu().detach() for f in fs_list]

        bs = ft_list[0].shape[0]
        weights = weights if weights is not None and len(weights) == len(ft_list) else [1.0] * len(ft_list)
        total_weight = sum(weights) if amap_mode == 'add' else 1.0 # Avoid division by zero if weights sum to 0

        anomaly_map = np.ones([bs, *out_size], dtype=np.float32) if amap_mode == 'mul' else np.zeros([bs, *out_size], dtype=np.float32)
        a_map_list = []

        if uni_am:
            # Upsample and normalize all features to the size of the first feature map
            target_size = (ft_list[0].shape[2], ft_list[0].shape[3])
            processed_ft = []
            processed_fs = []
            for ft, fs in zip(ft_list, fs_list):
                ft_norm = F.normalize(ft, p=2, dim=1)
                fs_norm = F.normalize(fs, p=2, dim=1)
                processed_ft.append(F.interpolate(ft_norm, size=target_size, mode='bilinear', align_corners=False))
                processed_fs.append(F.interpolate(fs_norm, size=target_size, mode='bilinear', align_corners=False))

            ft_map = torch.cat(processed_ft, dim=1)
            fs_map = torch.cat(processed_fs, dim=1)

            if use_cos:
                # Cosine similarity returns values in [-1, 1]. Distance is 1 - sim, so [0, 2].
                a_map = 1.0 - F.cosine_similarity(ft_map, fs_map, dim=1, eps=1e-6)
                a_map = a_map.unsqueeze(1) # Add channel dim for interpolate
            else:
                # L2 distance
                a_map = torch.sqrt(torch.sum((ft_map - fs_map) ** 2, dim=1, keepdim=True) + 1e-6)

            # Resize to final output size
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
            a_map = a_map.squeeze(1).numpy() # Remove channel dim, convert to numpy
            anomaly_map = a_map # Overwrite the initial map
            a_map_list.append(a_map) # Only one map in uni_am mode

        else: # Process each feature level separately
            for i in range(len(ft_list)):
                ft = ft_list[i]
                fs = fs_list[i]
                ft_norm = F.normalize(ft, p=2, dim=1)
                fs_norm = F.normalize(fs, p=2, dim=1)

                if use_cos:
                    a_map_i = 1.0 - F.cosine_similarity(ft_norm, fs_norm, dim=1, eps=1e-6)
                    a_map_i = a_map_i.unsqueeze(1)
                else:
                    a_map_i = torch.sqrt(torch.sum((ft_norm - fs_norm) ** 2, dim=1, keepdim=True) + 1e-6)

                # Resize individual map
                a_map_i = F.interpolate(a_map_i, size=out_size, mode='bilinear', align_corners=False)
                a_map_i = a_map_i.squeeze(1).numpy()
                a_map_list.append(a_map_i)

                if amap_mode == 'add':
                    anomaly_map += a_map_i * weights[i]
                elif amap_mode == 'mul':
                    anomaly_map *= np.maximum(a_map_i, 1e-6) # Prevent multiplying by zero if possible


            if amap_mode == 'add' and total_weight > 0:
                anomaly_map /= total_weight

        # Apply Gaussian smoothing to the final combined map
        if gaussian_sigma > 0:
            for idx in range(anomaly_map.shape[0]):
                anomaly_map[idx] = gaussian_filter(anomaly_map[idx], sigma=gaussian_sigma)

        return anomaly_map, a_map_list

    @staticmethod
    def _cal_pro_thr_worker(q, th, amaps_shared, masks_shared, idx_range):
        """Worker function for multiprocessing PRO calculation."""
        # Access shared memory arrays
        amaps = np.frombuffer(amaps_shared.get_obj(), dtype=np.float32).reshape(masks_shared.shape) # Assuming float32, adjust if needed
        masks = np.frombuffer(masks_shared.get_obj(), dtype=np.bool_).reshape(masks_shared.shape)

        binary_amaps_local = np.zeros_like(amaps[idx_range], dtype=bool)
        slice_amaps = amaps[idx_range]
        slice_masks = masks[idx_range]

        binary_amaps_local[slice_amaps <= th] = False
        binary_amaps_local[slice_amaps > th] = True

        pro_local = []
        for binary_amap, mask in zip(binary_amaps_local, slice_masks):
            # Skip if mask is all False (no anomaly regions)
            if not mask.any():
                continue
            labeled_mask = measure.label(mask)
            for region in measure.regionprops(labeled_mask):
                # Ensure coordinates are valid (sometimes regionprops might yield empty regions?)
                if region.area == 0:
                    continue
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pro_local.append(tp_pixels / region.area)

        # Calculate FP pixels within the slice for FPR calculation
        inverse_masks_local = np.logical_not(slice_masks)
        fp_pixels_local = np.logical_and(inverse_masks_local, binary_amaps_local).sum()
        inverse_mask_area_local = inverse_masks_local.sum()

        # Put results into the queue
        q.put({
            'th': th,
            'pro_list': pro_local,
            'fp_pixels': fp_pixels_local,
            'inverse_mask_area': inverse_mask_area_local
        })

    @staticmethod
    def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3, mp=False):
        """
        Calculates the Per-Region Overlap (PRO) score integrated up to a
        certain False Positive Rate (FPR).

        Args:
            masks (np.ndarray): Ground truth masks (N, H, W), boolean or 0/1.
            amaps (np.ndarray): Anomaly maps (N, H, W), float.
            max_step (int): Number of thresholds to evaluate.
            expect_fpr (float): Maximum FPR for AUC calculation.
            mp (bool): Use multiprocessing. (Can be slow due to overhead).

        Returns:
            float: The calculated PRO-AUC score.
        """
        if masks.max() == 0: # No anomaly regions exist
             logger.warning("PRO score calculation skipped: No positive regions in ground truth masks.")
             return 0.0

        masks = masks.astype(bool) # Ensure boolean type
        min_th, max_th = amaps.min(), amaps.max()
        if abs(max_th - min_th) < 1e-6:
             logger.warning("PRO score calculation skipped: Anomaly map is constant.")
             return 0.0 # Avoid division by zero delta

        delta = (max_th - min_th) / max_step
        thresholds = np.arange(min_th, max_th, delta)
        if len(thresholds) == 0:
            thresholds = np.array([min_th]) # Handle edge case if max_step is too small

        pros_all, fprs_all = [], []

        if mp and len(thresholds) > 1:
             # --- Multiprocessing Approach ---
             # Note: Significant overhead per threshold. Might be slower than serial for small datasets.
             logger.info(f"Calculating PRO score using multiprocessing with {len(thresholds)} thresholds...")
             # Create shared memory arrays
             N, H, W = amaps.shape
             amaps_shared = multiprocessing.Array('f', N * H * W) # 'f' for float
             masks_shared = multiprocessing.Array('b', N * H * W) # 'b' for ctypes bool (signed char) - check mapping

             # Wrap as numpy arrays to allow easy access/modification
             amaps_np = np.frombuffer(amaps_shared.get_obj(), dtype=np.float32).reshape(N, H, W)
             masks_np = np.frombuffer(masks_shared.get_obj(), dtype=np.bool_).reshape(N, H, W) # Read back as bool

             # Copy data into shared memory
             np.copyto(amaps_np, amaps.astype(np.float32))
             np.copyto(masks_np, masks) # Should handle bool correctly

             num_processes = min(multiprocessing.cpu_count(), len(thresholds)) # Limit processes
             q = multiprocessing.Manager().Queue()
             processes = []

             # Simple static distribution of thresholds to processes (could be improved)
             thresholds_per_proc = np.array_split(thresholds, num_processes)
             proc_idx_ranges = [(0, N)] * num_processes # Each process handles all images for its thresholds

             for i in range(num_processes):
                  for th in thresholds_per_proc[i]:
                       p = multiprocessing.Process(
                           target=Evaluator._cal_pro_thr_worker,
                           args=(q, th, amaps_shared, masks_shared, proc_idx_ranges[i])
                       )
                       processes.append(p)
                       p.start()

             # Collect results
             results_list = []
             for _ in range(len(thresholds)):
                 try:
                     # Add a timeout to prevent hanging indefinitely
                     results_list.append(q.get(timeout=300)) # 5 minutes timeout per threshold
                 except multiprocessing.queues.Empty:
                     logger.error("Timeout occurred waiting for PRO worker result. Aborting PRO calculation.")
                     # Terminate running processes
                     for p in processes:
                         if p.is_alive():
                             p.terminate()
                             p.join(timeout=1) # Wait briefly for termination
                     return 0.0 # Indicate failure


             for p in processes:
                 p.join()

             # Process collected results - aggregate stats per threshold
             results_by_th = {}
             for res in results_list:
                  th = res['th']
                  if th not in results_by_th:
                       results_by_th[th] = {'pro_list': [], 'fp_pixels': 0, 'inverse_mask_area': 0}
                  results_by_th[th]['pro_list'].extend(res['pro_list'])
                  results_by_th[th]['fp_pixels'] += res['fp_pixels']
                  results_by_th[th]['inverse_mask_area'] += res['inverse_mask_area']

             # Calculate final PRO and FPR for each threshold
             total_inverse_mask_area = np.logical_not(masks).sum()
             if total_inverse_mask_area == 0:
                 logger.warning("PRO score: Total inverse mask area is zero. FPR cannot be calculated.")
                 return 0.0

             th_fpr_pro = []
             for th, data in results_by_th.items():
                 fpr = data['fp_pixels'] / total_inverse_mask_area if total_inverse_mask_area > 0 else 0
                 pro = np.mean(data['pro_list']) if data['pro_list'] else 0
                 th_fpr_pro.append((th, fpr, pro))

             # Sort by threshold
             th_fpr_pro.sort(key=lambda x: x[0])
             ths = [x[0] for x in th_fpr_pro]
             fprs_all = [x[1] for x in th_fpr_pro]
             pros_all = [x[2] for x in th_fpr_pro]

        else:
            # --- Serial Calculation ---
            logger.info(f"Calculating PRO score serially with {len(thresholds)} thresholds...")
            binary_amaps = np.zeros_like(amaps, dtype=bool)
            total_inverse_mask_area = np.logical_not(masks).sum()

            if total_inverse_mask_area == 0:
                 logger.warning("PRO score: Total inverse mask area is zero. FPR cannot be calculated.")
                 return 0.0

            for th in thresholds:
                binary_amaps[amaps <= th] = False
                binary_amaps[amaps > th] = True

                pro_current_th = []
                fp_pixels_current_th = 0
                # Iterate through each image/mask pair
                for i in range(masks.shape[0]):
                    mask_i = masks[i]
                    binary_amap_i = binary_amaps[i]

                    # Calculate PRO for this image
                    if mask_i.any(): # Only if there are anomaly regions
                        labeled_mask_i = measure.label(mask_i)
                        for region in measure.regionprops(labeled_mask_i):
                            if region.area == 0: continue
                            coords = region.coords
                            tp_pixels = binary_amap_i[coords[:, 0], coords[:, 1]].sum()
                            pro_current_th.append(tp_pixels / region.area)

                    # Calculate FP pixels for this image
                    inverse_mask_i = np.logical_not(mask_i)
                    fp_pixels_current_th += np.logical_and(inverse_mask_i, binary_amap_i).sum()

                # Calculate average PRO and FPR for this threshold
                pro = np.mean(pro_current_th) if pro_current_th else 0.0
                fpr = fp_pixels_current_th / total_inverse_mask_area
                pros_all.append(pro)
                fprs_all.append(fpr)

        # --- Calculate AUC ---
        pros = np.array(pros_all)
        fprs = np.array(fprs_all)

        # Filter by expected FPR
        idxes = fprs <= expect_fpr
        if not np.any(idxes):
            logger.warning(f"No FPR values below the threshold {expect_fpr}. PRO-AUC will be 0.")
            return 0.0

        fprs_filtered = fprs[idxes]
        pros_filtered = pros[idxes]

        # Ensure FPRs are monotonically increasing for AUC calculation
        # Sort by FPR, then threshold (implicit via threshold sort earlier)
        sort_indices = np.argsort(fprs_filtered)
        fprs_sorted = fprs_filtered[sort_indices]
        pros_sorted = pros_filtered[sort_indices]

        # Remove duplicates in FPRs, keeping the highest PRO for that FPR
        unique_fprs, unique_indices = np.unique(fprs_sorted, return_index=True)
        unique_pros = np.maximum.accumulate(pros_sorted[::-1])[::-1][unique_indices] # Get max PRO for each unique FPR

        # Normalize FPRs to [0, 1] range within the filtered set for AUC stability
        fprs_norm = unique_fprs # Use the actual FPR values up to expect_fpr

        if len(fprs_norm) < 2:
            logger.warning(f"Not enough points ({len(fprs_norm)}) after filtering/uniqueing for PRO-AUC calculation.")
            return 0.0

        # Calculate AUC using trapezoidal rule
        pro_auc = auc(fprs_norm, unique_pros)


        return pro_auc

