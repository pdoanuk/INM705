# -*- coding: utf-8 -*-
## TIMESTAMP @ 2025-04-23T09:00:45
## author: phuocddat
## start
# Adapted and improved from original ViTAD & Timm Vision Transformer

## end --
import time
import logging
import math
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import VisionTransformer, Block, Attention, Mlp, _cfg, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== Utility Functions ==========

def get_timepc() -> float:
    """Returns high-resolution time, synchronizing CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def get_net_params(net: nn.Module) -> float:
    """Calculates the number of trainable parameters in a network (in millions)."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6


# ========== Custom Vision Transformer Components ==========

class ViTEncoder(VisionTransformer):
    """
    Modified VisionTransformer to act as an encoder, outputting features
    from specified intermediate 'teacher' layers and final 'neck' layers.
    """
    def __init__(
        self,
        teacher_layer_indices: List[int],
        neck_layer_indices: List[int],
        *args, **kwargs
    ):
        """
        Args:
            teacher_layer_indices (List[int]): 1-based indices of Transformer blocks
                                               whose outputs should be returned as
                                               'teacher' features (B, C, H, W).
            neck_layer_indices (List[int]): 1-based indices of Transformer blocks
                                            whose outputs should be returned as
                                            'neck' features (B, L, C) for fusion.
            *args, **kwargs: Arguments passed to timm's VisionTransformer.
                             'num_classes' should typically be 0 for feature extraction.
        """
        # Ensure num_classes=0 unless explicitly overridden for feature extraction
        kwargs.setdefault('num_classes', 0)
        if kwargs['num_classes'] > 0:
            logger.warning(f"Instantiating ViTEncoder with num_classes={kwargs['num_classes']}. "
                           "The head classification layer will be present but likely unused.")

        super().__init__(*args, **kwargs)

        if not isinstance(teacher_layer_indices, (list, tuple)) or not all(isinstance(i, int) for i in teacher_layer_indices):
             raise ValueError(f"teacher_layer_indices must be a list/tuple of integers, got {teacher_layer_indices}")
        if not isinstance(neck_layer_indices, (list, tuple)) or not all(isinstance(i, int) for i in neck_layer_indices):
             raise ValueError(f"neck_layer_indices must be a list/tuple of integers, got {neck_layer_indices}")

        max_index = max(teacher_layer_indices + neck_layer_indices) if (teacher_layer_indices + neck_layer_indices) else 0
        if max_index > len(self.blocks):
            raise ValueError(f"Maximum layer index ({max_index}) exceeds number of blocks ({len(self.blocks)})")
        if any(i <= 0 for i in teacher_layer_indices + neck_layer_indices):
            raise ValueError("Layer indices must be positive (1-based).")

        self.teacher_layer_indices = set(teacher_layer_indices)
        self.neck_layer_indices = set(neck_layer_indices)

        # Remove the classification head if num_classes is 0
        if self.num_classes == 0:
             self.head = nn.Identity()
             self.head_dist = nn.Identity() # For DeiT distil token
             self.fc_norm = nn.Identity() # Norm layer before head

        logger.info(f"ViTEncoder initialized. Teacher layers: {sorted(list(self.teacher_layer_indices))}, "
                    f"Neck layers: {sorted(list(self.neck_layer_indices))}. Num blocks: {len(self.blocks)}")

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H_in, W_in).

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - teacher_features: List of features (B, C, H_grid, W_grid) from specified teacher layers.
                - neck_features: List of features (B, L, C) from specified neck layers (excluding prefix tokens).
        """
        B = x.shape[0]
        x = self.patch_embed(x) # B, L, C (L = H*W + num_prefix_tokens)
        x = self._pos_embed(x) # Adds positional encoding + class token etc.
        x = self.norm_pre(x) # Optional pre-normalization

        teacher_features = []
        neck_features = []

        # Grid size calculation (needed for reshaping teacher features)
        # Handle potential dynamic image size if patch_embed supports it
        if hasattr(self.patch_embed, 'dynamic_img_size') and self.patch_embed.dynamic_img_size:
            H_grid, W_grid = self.patch_embed.get_grid_size(x.shape[:2]) # Requires timm >= 0.5.5
        else:
            H_grid, W_grid = self.patch_embed.grid_size
        num_patches = H_grid * W_grid

        current_input = x
        for i, blk in enumerate(self.blocks):
            # Apply block (handles grad checkpointing internally if enabled)
            x = blk(current_input)
            layer_idx = i + 1 # 1-based index

            # Extract features *after* the block
            # Exclude prefix tokens (e.g., cls token) for spatial features
            spatial_features = x[:, self.num_prefix_tokens:] # B, L_spatial, C where L_spatial = H_grid * W_grid

            if spatial_features.shape[1] != num_patches:
                 logger.warning(
                    f"Layer {layer_idx}: Spatial feature sequence length ({spatial_features.shape[1]}) "
                    f"doesn't match expected number of patches ({num_patches}). "
                 )

            # Collect neck features (B, L_spatial, C)
            if layer_idx in self.neck_layer_indices:
                neck_features.append(spatial_features)

            if layer_idx in self.teacher_layer_indices:
                if spatial_features.shape[1] == num_patches:
                    teacher_feat = spatial_features.view(B, H_grid, W_grid, self.embed_dim).permute(0, 3, 1, 2).contiguous()
                    teacher_features.append(teacher_feat)
                else:
                     logger.error(f"Cannot reshape teacher features at layer {layer_idx} due to length mismatch: "
                                  f"Got {spatial_features.shape[1]}, expected {num_patches}. Skipping teacher output.")

            current_input = x


        return teacher_features, neck_features


class ViTDecoder(nn.Module):
    """
    A Transformer-based decoder operating on sequence features.
    It processes features from the encoder/fusion layers and outputs
    features from specified intermediate 'student' layers.
    """
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        act_layer: nn.Module = nn.GELU,
        student_layer_indices: List[int] = [],
    ):
        """
        Args:
            num_patches (int): The sequence length (L) of the input features.
                               This determines the size of the positional embedding.
            embed_dim (int): Dimension of the input features (C).
            depth (int): Number of transformer blocks in the decoder.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio for MLP hidden dimension.
            qkv_bias (bool): Whether to include bias in QKV projections.
            qk_scale (Optional[float]): Override default QK scale.
            drop_rate (float): Dropout rate for positional embedding.
            attn_drop_rate (float): Dropout rate for attention weights.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer constructor.
            act_layer (nn.Module): Activation layer constructor.
            student_layer_indices (List[int]): 1-based indices of Transformer blocks
                                               whose outputs should be returned as
                                               'student' features (B, C, H, W).
        """
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.depth = depth
        self.student_layer_indices = set(student_layer_indices)

        if not isinstance(student_layer_indices, (list, tuple)) or not all(isinstance(i, int) for i in student_layer_indices):
             raise ValueError(f"student_layer_indices must be a list/tuple of integers, got {student_layer_indices}")
        max_index = max(student_layer_indices) if student_layer_indices else 0
        if max_index > depth:
            raise ValueError(f"Maximum student layer index ({max_index}) exceeds decoder depth ({depth})")
        if any(i <= 0 for i in student_layer_indices):
            raise ValueError("Layer indices must be positive (1-based).")


        # Positional embedding for the input sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        trunc_normal_(self.pos_embed, std=.02)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer
            )
            for i in range(depth)])

        # Final normalization layer (applied after the last block if needed, but often outputs are taken before)
        self.norm = norm_layer(embed_dim)

        logger.info(f"ViTDecoder initialized. Input L={num_patches}, C={embed_dim}, Depth={depth}. "
                    f"Student layers: {sorted(list(self.student_layer_indices))}")


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input sequence features (B, L, C), where L = num_patches.

        Returns:
            List[torch.Tensor]: List of features (B, C, H, W) from specified student layers.
                                The spatial dimensions (H, W) are inferred assuming a square grid (H*W=L).
        """
        B, L, C = x.shape
        if L != self.num_patches:
            raise ValueError(f"Input sequence length ({L}) does not match decoder's expected num_patches ({self.num_patches}).")
        if C != self.embed_dim:
            raise ValueError(f"Input feature dimension ({C}) does not match decoder's embed_dim ({self.embed_dim}).")

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        student_features = []
        H = W = int(math.sqrt(self.num_patches))

        # Process through decoder blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            layer_idx = i + 1 # 1-based index

            if layer_idx in self.student_layer_indices:
                # Output after the block, before the final norm
                feature_out = x # B, L, C
                # Reshape to B, C, H, W
                feature_out = feature_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

                student_features.append(feature_out)


        return student_features


class Fusion(nn.Module):
    """
    Fuses a list of feature tensors by concatenation and a linear projection.
    """
    def __init__(self, embed_dim: int, num_features_to_fuse: int):
        """
        Args:
            embed_dim (int): The embedding dimension (C) of the input features
                             and the desired output dimension.
            num_features_to_fuse (int): The number of feature tensors in the input list.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_features_to_fuse = num_features_to_fuse
        self.expected_input_dim = embed_dim * num_features_to_fuse

        # Linear layer to project concatenated features back to embed_dim
        self.fc = nn.Linear(self.expected_input_dim, embed_dim)

        logger.info(f"Fusion module initialized. Fusing {num_features_to_fuse} features. "
                    f"Input dim: {self.expected_input_dim}, Output dim: {embed_dim}")

    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Fused features (B, L, C).
        """
        if isinstance(features, list):
            # Concatenate along the channel dimension (C)
            feature_combined = torch.cat(features, dim=2) # B, L, C * num_features

        elif isinstance(features, torch.Tensor):
             feature_combined = features # B, L, C
        else:
             raise TypeError(f"Fusion input must be a list of tensors or a single tensor, got {type(features)}")

        # Apply linear projection
        fused_output = self.fc(feature_combined) # B, L, C
        return fused_output


# ========== Model Creator Functions ==========

def _create_vit_encoder(variant: str, pretrained: bool = False, **kwargs) -> ViTEncoder:
    """Creates a ViTEncoder instance using timm's build_model_with_cfg."""
    logger.info(f"Creating ViTEncoder: {variant}, pretrained={pretrained}")
    # Extract encoder-specific args
    teacher_layers = kwargs.pop('teacher_layer_indices', [])
    neck_layers = kwargs.pop('neck_layer_indices', [])

    if not teacher_layers and not neck_layers:
         logger.warning(f"ViTEncoder {variant} created with no teacher or neck layers specified. "
                        "It will run but produce empty feature lists.")

    # Check for incompatible kwargs (features_only is handled implicitly by ViTEncoder)
    if kwargs.pop('features_only', None):
        logger.warning("'features_only' kwarg is ignored for ViTEncoder.")

    # Handle checkpoint filtering for flexible input sizes ('flexi' models)
    if 'flexi' in variant:
        try:
            _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=True)
        except TypeError:
            _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear')
    else:
        _filter_fn = checkpoint_filter_fn

    try:
        model = build_model_with_cfg(
            ViTEncoder,
            variant,
            pretrained=pretrained,
            pretrained_filter_fn=_filter_fn,
            teacher_layer_indices=teacher_layers,
            neck_layer_indices=neck_layers,
            **kwargs
        )
        logger.info(f"Successfully created/loaded ViTEncoder: {variant}")
        return model
    except Exception as e:
        logger.error(f'Failed to create/load ViTEncoder {variant}. Error: {e}')


def _create_vit_decoder(base_kwargs: Dict[str, Any], **kwargs) -> ViTDecoder:
    """Creates a ViTDecoder instance."""
    logger.info(f"Creating ViTDecoder...")
    model_kwargs = base_kwargs.copy()
    model_kwargs.update(kwargs)

    img_size = model_kwargs.pop('img_size', None) # Used to calculate num_patches if not provided
    patch_size = model_kwargs.pop('patch_size', None) # Used to calculate num_patches if not provided
    num_patches_arg = model_kwargs.pop('num_patches', None) # Allow direct override

    if num_patches_arg is not None:
        num_patches = num_patches_arg
        logger.info(f"Using provided 'num_patches': {num_patches}")
    elif img_size is not None and patch_size is not None:
         if isinstance(img_size, int):
             img_size = (img_size, img_size)
         if isinstance(patch_size, int):
              patch_size = (patch_size, patch_size)
         grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
         num_patches = grid_size[0] * grid_size[1]
         logger.info(f"Calculated 'num_patches' from img_size={img_size}, patch_size={patch_size}: {num_patches}")
    else:
         raise ValueError(f"Error 'num_patches' from img_size={img_size}, patch_size={patch_size}")

    # Ensure required args for ViTDecoder are present
    required_args = ['embed_dim', 'depth', 'num_heads', 'student_layer_indices']
    for arg in required_args:
         if arg not in model_kwargs:
              raise ValueError(f"Missing required argument '{arg}' for ViTDecoder.")

    # Instantiate the decoder
    model = ViTDecoder(
        num_patches=num_patches,
        embed_dim=model_kwargs['embed_dim'],
        depth=model_kwargs['depth'],
        num_heads=model_kwargs['num_heads'],
        mlp_ratio=model_kwargs.get('mlp_ratio', 4.0),
        qkv_bias=model_kwargs.get('qkv_bias', True),
        qk_scale=model_kwargs.get('qk_scale', None),
        drop_rate=model_kwargs.get('drop_rate', 0.0),
        attn_drop_rate=model_kwargs.get('attn_drop_rate', 0.0),
        drop_path_rate=model_kwargs.get('drop_path_rate', 0.0),
        norm_layer=model_kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
        act_layer=model_kwargs.get('act_layer', nn.GELU),
        student_layer_indices=model_kwargs['student_layer_indices']
    )
    logger.info(f"Successfully created ViTDecoder.")
    return model


# --- Encoders ---
def vit_small_patch16_224_dino(pretrained=True, **kwargs) -> ViTEncoder:
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model_kwargs.update(kwargs)
    return _create_vit_encoder('vit_small_patch16_224.dino', pretrained=pretrained, **model_kwargs)

def vit_base_patch16_224(pretrained=True, **kwargs) -> ViTEncoder:
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model_kwargs.update(kwargs)
    return _create_vit_encoder('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=pretrained, **model_kwargs)

# --- Decoders ---
def de_vit_small_patch16_base(
    img_size: int,
    patch_size: int = 16,
    depth: int = 6, # Example depth, configure as needed
    student_layer_indices: Optional[List[int]] = None,
    **kwargs
) -> ViTDecoder:
    """ Base configuration for a 'small' ViT decoder. """
    base_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=384, # Matches vit_small embed_dim
        num_heads=6, # Matches vit_small num_heads
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        student_layer_indices=student_layer_indices if student_layer_indices is not None else []
    )
    # Override defaults and add depth
    base_kwargs.update(kwargs)
    base_kwargs['depth'] = depth
    return _create_vit_decoder(base_kwargs)

# --- Fusion ---
def fusion_module(embed_dim: int, num_features_to_fuse: int, **kwargs) -> Fusion:
     """ Creates the Fusion module. """
     return Fusion(embed_dim=embed_dim, num_features_to_fuse=num_features_to_fuse)


MODEL_CREATORS: Dict[str, callable] = {
    # Fusion
    'fusion': fusion_module,
    # Encoders (add more timm variants or custom ones here)
    'vit_small_patch16_224_dino': vit_small_patch16_224_dino,
    'vit_base_patch16_224': vit_base_patch16_224,
    # Decoders (define configurations as needed)
    'de_vit_small_patch16_base': de_vit_small_patch16_base,

}


# ========== Helper for Model Creation and Loading ==========

def create_and_load_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """
    Creates a model component using MODEL_CREATORS and handles loading weights.

    """
    model_name = model_cfg.get('name')
    if not model_name:
        raise ValueError("Model configuration must include a 'name'.")

    kwargs = model_cfg.get('kwargs', {}).copy() # Work on a copy

    if model_name not in MODEL_CREATORS:
        raise ValueError(f"Unknown model name: '{model_name}'. Available models: {list(MODEL_CREATORS.keys())}")

    model_creator_fn = MODEL_CREATORS[model_name]

    use_timm_pretrained = kwargs.pop('pretrained', False)
    checkpoint_path = kwargs.pop('checkpoint_path', '')
    strict_loading = kwargs.pop('strict', True)

    # --- Model Instantiation ---
    creator_kwargs = kwargs
    if checkpoint_path:
        logger.warning(f"Checkpoint path '{checkpoint_path}' provided for {model_name}. "
                       f"Ignoring 'pretrained={use_timm_pretrained}' flag.")
        creator_kwargs['pretrained'] = False  # Prevent timm from loading standard weights
    else:
        creator_kwargs['pretrained'] = use_timm_pretrained

    logger.info(f"Creating model component: '{model_name}'")
    logger.debug(f"Creator function kwargs: {creator_kwargs}")
    model = model_creator_fn(**creator_kwargs)

    # --- Custom Checkpoint Loading ---
    if checkpoint_path:
        logger.info(f"Attempting to load checkpoint for '{model_name}' from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')

            # Find the state dict within the checkpoint file
            state_dict = None
            if isinstance(ckpt, dict):
                # Common keys for state_dict
                potential_keys = ['state_dict', 'model', 'net', 'module']
                for key in potential_keys:
                    if key in ckpt:
                        state_dict = ckpt[key]
                        logger.info(f"Found state_dict under key '{key}' in checkpoint.")
                        break
                if state_dict is None:
                    logger.warning(f"Could not find a standard state_dict key in the checkpoint dict for {model_name}. "
                                   f"Assuming the checkpoint dict *is* the state_dict.")
                    state_dict = ckpt # Assume the whole dict is the state_dict
            elif isinstance(ckpt, nn.Module):
                 # Sometimes the entire model is saved
                 logger.warning(f"Checkpoint for {model_name} seems to be a full model instance. Extracting state_dict.")
                 state_dict = ckpt.state_dict()
            elif isinstance(ckpt, dict): # Should have been caught above, but double check
                 state_dict = ckpt
            else:
                 raise TypeError(f"Checkpoint for {model_name} is not a dictionary or nn.Module. Type: {type(ckpt)}")

            if not isinstance(state_dict, dict):
                raise ValueError(f"Could not extract a valid state dictionary (dict) from checkpoint {checkpoint_path}")

            # --- State Dict Loading into Model ---
            if hasattr(model, 'load_state_dict'):
                logger.info(f"Loading state_dict into '{model_name}' with strict={strict_loading}")
                try:
                    new_state_dict = {}
                    for k, v in state_dict.items():
                         name = k[7:] if k.startswith('module.') else k # remove `module.` prefix
                         new_state_dict[name] = v
                    state_dict = new_state_dict

                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict_loading)

                    if missing_keys:
                        logger.warning(f"Missing keys when loading checkpoint for {model_name} (strict={strict_loading}): {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys when loading checkpoint for {model_name} (strict={strict_loading}): {unexpected_keys}")
                    if not missing_keys and not unexpected_keys:
                         logger.info(f"Successfully loaded state_dict for {model_name}.")

                except Exception as e:
                    logger.error(f"Error during state_dict loading for {model_name} (strict={strict_loading}): {e}", exc_info=True)
                    if strict_loading:
                        logger.warning("Attempting to load state_dict with strict=False...")
                        try:
                             missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                             logger.info(f"Successfully loaded state_dict for {model_name} with strict=False.")
                             if missing_keys: logger.warning(f"  Missing keys (strict=False): {missing_keys}")
                             if unexpected_keys: logger.warning(f"  Unexpected keys (strict=False): {unexpected_keys}")
                        except Exception as e2:
                             logger.error(f"Failed to load state_dict for {model_name} even with strict=False. Error: {e2}", exc_info=True)
                             raise e from e2
                    else:
                        raise e
            else:
                logger.error(f"Model '{model_name}' does not have a 'load_state_dict' method. Cannot load checkpoint.")

        except FileNotFoundError:
            logger.error(f"Checkpoint file not found at: {checkpoint_path}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred loading checkpoint for {model_name} from {checkpoint_path}: {e}", exc_info=True)
            raise

    elif use_timm_pretrained and 'pretrained' not in model_creator_fn.__code__.co_varnames:
        # If pretrained=True was set but the creator doesn't handle it (e.g., Fusion, Decoder)
        logger.warning(f"Model '{model_name}' does not support standard 'pretrained' weights. Flag ignored.")


    return model


# ========== ViTAD_v2 Main Model ==========
class ViTAD_v2(nn.Module):
    """
    Vision Transformer for Anomaly Detection (ViTAD) model.

    Enhanced version
    """
    def __init__(self, model_t_cfg: Dict[str, Any], model_f_cfg: Dict[str, Any], model_s_cfg: Dict[str, Any]):
        """
        Initializes the ViTAD_v2 model by creating its components.

        Args:
            model_t_cfg (dict): Configuration dictionary for the teacher model (ViTEncoder).
                                Must specify 'teacher_layer_indices' and 'neck_layer_indices' in kwargs.
            model_f_cfg (dict): Configuration dictionary for the fusion model (Fusion).
                                Must specify 'embed_dim' and 'num_features_to_fuse' in kwargs.
                                'num_features_to_fuse' must match the length of teacher's 'neck_layer_indices'.
            model_s_cfg (dict): Configuration dictionary for the student model (ViTDecoder).
                                Must specify 'student_layer_indices', 'embed_dim', 'depth',
                                'num_heads', and ('num_patches' or 'img_size'/'patch_size') in kwargs.
                                'embed_dim' should match fusion output dim.
                                'num_patches' should match encoder's grid size.
        """
        super().__init__()

        logger.info("--- Initializing ViTAD v2 ---")
        logger.debug(f"Teacher Config: {model_t_cfg}")
        logger.debug(f"Fusion Config: {model_f_cfg}")
        logger.debug(f"Student Config: {model_s_cfg}")

        # --- Validate Configurations ---
        self._validate_configs(model_t_cfg, model_f_cfg, model_s_cfg)

        # --- Create Sub-models ---
        logger.info("Creating Teacher Network (net_t)...")
        self.net_t = create_and_load_model(model_t_cfg)

        logger.info("Creating Fusion Network (net_fusion)...")
        self.net_fusion = create_and_load_model(model_f_cfg)

        logger.info("Creating Student Network (net_s)...")
        self.net_s = create_and_load_model(model_s_cfg)

        # Freeze the teacher network
        self.freeze_teacher()

        logger.info("--- ViTAD_v2 Initialization Complete ---")
        logger.info(f"Teacher params (M): {get_net_params(self.net_t):.2f} (Frozen)")
        logger.info(f"Fusion params (M): {get_net_params(self.net_fusion):.2f}")
        logger.info(f"Student params (M): {get_net_params(self.net_s):.2f}")


    def freeze_teacher(self):
        """ Freezes the teacher network (sets to eval mode, disables gradients). """
        logger.info("Freezing Teacher Network (net_t)...")
        self.net_t.eval()
        for param in self.net_t.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """ Sets the training mode for the ViTAD_v2 model. Teacher remains frozen. """
        self.training = mode
        # Teacher is always frozen and in eval mode
        self.net_t.eval()
        # Set mode for fusion and student networks
        self.net_fusion.train(mode)
        self.net_s.train(mode)
        logger.debug(f"ViTAD_v2 set to {'train' if mode else 'eval'} mode. Teacher remains frozen.")
        return self

    def eval(self):
        """ Sets the ViTAD_v2 model to evaluation mode. """
        return self.train(False)

    def forward(self, imgs: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the ViTAD_v2 model.

        Args:
            imgs (torch.Tensor): Input image batch (B, C, H, W).

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - teacher_features (List[torch.Tensor]): Output features from the frozen teacher's
                                                          specified layers (B, C, H_grid, W_grid).
                - student_features (List[torch.Tensor]): Output features from the student decoder's
                                                          specified layers (B, C, H_grid, W_grid or B, L, C).
        """
        # --- Teacher Forward Pass (Frozen) ---
        with torch.no_grad():
            feats_t, feats_n = self.net_t(imgs)

        # feats_n_detached = [f.detach() for f in feats_n] # Not strictly needed with no_grad context

        # --- Fusion Forward Pass ---
        # Fuses the neck features: List[B, L, C] -> B, L, C

        fused_features = self.net_fusion(feats_n) # Use original feats_n

        # --- Student Forward Pass ---
        # Input: B, L, C -> Output: List[B, C, H_grid, W_grid] (or List[B, L, C])

        feats_s = self.net_s(fused_features)

        # Return teacher features (targets) and student features (predictions)
        return feats_t, feats_s


# ========== Default Model Loading Example ==========

def load_default_vitad_model(device: Optional[Union[str, torch.device]] = None) -> ViTAD_v2:
    """
    Instantiates the ViTAD_v2 model with a default configuration
    (e.g., ViT-Small DINO teacher, corresponding decoder).

    Args:
        device (Optional[Union[str, torch.device]]): Device to move the model to ('cpu', 'cuda', etc.).

    Returns:
        ViTAD: The instantiated ViTAD_v2 model.
    """
    image_size = 256  # Example image size
    patch_size = 16   # Must match teacher's patch size
    teacher_embed_dim = 384 # For ViT-Small
    teacher_depth = 12
    num_heads = 6

    # Layer indices (1-based)
    teacher_layers = [3, 6, 9] # Layers for reconstruction loss
    neck_layers = [12]        # Final layer output(s) for fusion -> student input
    student_layers = [3, 6, 9] # Layers matching teacher layers for loss calculation
    decoder_depth = 9         # Depth of the student decoder

    # Calculate num_patches based on encoder's grid
    num_patches = (image_size // patch_size) ** 2

    # --- Configuration Dictionaries ---
    model_t_cfg = {
        'name': 'vit_small_patch16_224_dino',
        'kwargs': dict(
            pretrained=True,
            checkpoint_path='',
            img_size=image_size,
            embed_dim=teacher_embed_dim,
            depth=teacher_depth,
            num_heads=num_heads,
            # Encoder-specific args:
            teacher_layer_indices=teacher_layers,
            neck_layer_indices=neck_layers,
        )
    }

    model_f_cfg = {
        'name': 'fusion',
        'kwargs': dict(
            checkpoint_path='',
            strict=True,
            embed_dim=teacher_embed_dim,
            num_features_to_fuse=len(neck_layers)
        )
    }

    model_s_cfg = {
        'name': 'de_vit_small_patch16_base',
        'kwargs': dict(
            checkpoint_path='',
            strict=True,
            embed_dim=teacher_embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            num_patches=num_patches,
            img_size=image_size,
            patch_size=patch_size,
            student_layer_indices=student_layers,
        )
    }

    logger.info("Instantiating ViTAD_v2 with default configuration...")
    try:
        net = ViTAD_v2(
            model_t_cfg=model_t_cfg,
            model_f_cfg=model_f_cfg,
            model_s_cfg=model_s_cfg
        )
        logger.info("ViTAD_v2 model instantiated successfully.")
        if device:
            net = net.to(device)
            logger.info(f"Moved ViTAD_v2 model to device: {device}")

    except Exception as e:
        logger.error(f"!!! Error during ViTAD_v2 instantiation: {e} !!!", exc_info=True)

        raise

    return net

## testing
if __name__ == "__main__":
    try:

        logging.getLogger().setLevel(logging.DEBUG)

        print("Testing default ViTAD_v2 model creation...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_default_vitad_model(device=device)
        print(f"Model loaded successfully on {device}.")
        print(f"Total parameters (Trainable): {get_net_params(model):.2f} M")

        # Test forward pass with dummy data
        print("Testing forward pass...")
        dummy_input = torch.randn(4, 3, 256, 256).to(device)
        model.eval() # Set to eval mode
        with torch.no_grad():
             teacher_out, student_out = model(dummy_input)

        print(f"Teacher output feature list length: {len(teacher_out)}")
        if teacher_out:
            print(f"  Shape of first teacher feature: {teacher_out[0].shape}")
        print(f"Student output feature list length: {len(student_out)}")
        if student_out:
            print(f"  Shape of first student feature: {student_out[0].shape}")

        print("Testing train mode switching...")
        model.train()
        print(f" net_t is training: {model.net_t.training}, requires_grad: {any(p.requires_grad for p in model.net_t.parameters())}")
        print(f" net_fusion is training: {model.net_fusion.training}, requires_grad: {any(p.requires_grad for p in model.net_fusion.parameters())}")
        print(f" net_s is training: {model.net_s.training}, requires_grad: {any(p.requires_grad for p in model.net_s.parameters())}")

        model.eval()
        print(f" net_t is training: {model.net_t.training}, requires_grad: {any(p.requires_grad for p in model.net_t.parameters())}")
        print(f" net_fusion is training: {model.net_fusion.training}, requires_grad: {any(p.requires_grad for p in model.net_fusion.parameters())}")
        print(f" net_s is training: {model.net_s.training}, requires_grad: {any(p.requires_grad for p in model.net_s.parameters())}")


    except Exception as main_e:
        print(f"\n--- An error occurred during the example run --- {main_e}")

        pass