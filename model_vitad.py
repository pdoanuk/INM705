# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn
from functools import partial

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from timm.models.vision_transformer import VisionTransformer, _cfg, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg

from timm.models.layers import trunc_normal_, set_layer_config

# ========== Utility Functions ==========

def get_timepc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def get_net_params(net):
    num_params = 0
    for param in net.parameters():
        if param.requires_grad:
            num_params += param.numel()
    return num_params / 1e6


# ========== Model Definitions ==========
def de_vit_small_patch16_224_dino(pretrained=False, **kwargs):
    base_kwargs = dict(patch_size=16, embed_dim=384, num_heads=6, mlp_ratio=4,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return _create_decoder(base_kwargs, **kwargs)

def vit_small_patch16_224_dino(pretrained=True, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model_kwargs.update(kwargs)
    return _create_vision_transformer('vit_small_patch16_224.dino', pretrained=pretrained, **model_kwargs)

def vit_base_patch16_224(pretrained=True, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model_kwargs.update(kwargs)
    return _create_vision_transformer('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=pretrained,
                                      **model_kwargs)

# ========== Fusion ==========
class Fusion(nn.Module):
    def __init__(self, dim, mul):
        super(Fusion, self).__init__()
        self.fc = nn.Linear(dim * mul, dim)

    def forward(self, features):
        # Input: list of features [B, L, C] (assuming L is same for all neck features)
        # If multiple neck features, ensure they have the same L dimension
        if isinstance(features, list):
            if len(features) > 1:
                # Ensure sequence lengths are compatible for concatenation
                assert all(f.shape[1] == features[0].shape[1] for f in features), \
                    "Sequence lengths of neck features must match for fusion."
                feature_align = torch.cat(features, dim=2)  # Concatenate along the channel dim
            else:
                feature_align = features[0]  # Only one feature, no need to cat
        else:  # If input is already a tensor (e.g., single neck feature)
            feature_align = features

        # Check if FC input dim matches concatenated feature dim
        expected_dim = self.fc.in_features
        actual_dim = feature_align.shape[2]
        if expected_dim != actual_dim:
            raise ValueError(
                f"Fusion FC layer expects input dim {expected_dim}, but got {actual_dim}. Check 'dim' and 'mul' config.")

        feature_align = self.fc(feature_align)  # Apply linear layer
        return feature_align


def fusion(pretrained=False, **kwargs):
    model = Fusion(**kwargs)
    return model


class ViT_Encoder(VisionTransformer):
    def __init__(self, teachers, neck, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teachers
        self.neck = neck

    def forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)  # timm's VisionTransformer uses _pos_embed method

        out_neck, out_t = [], []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            current_input = x
            for i, blk in enumerate(self.blocks):
                x = blk(current_input)
                fea = x[:, self.num_prefix_tokens:]
                if i + 1 in self.neck:
                    out_neck.append(fea)
                if i + 1 in self.teachers:
                    B_f, L_f, C_f = fea.shape
                    H = W = self.patch_embed.grid_size[0]  # Use grid size from patch_embed
                    if L_f != H * W: raise ValueError(f"Feature len {L_f} != grid H*W {H * W}")
                    fea_t = fea.view(B_f, H, W, C_f).permute(0, 3, 1, 2).contiguous()
                    out_t.append(fea_t)
                current_input = x

        else:
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                # Features excluding cls token(s)
                fea = x[:, self.num_prefix_tokens:]  # Use num_prefix_tokens (usually 1 for cls)
                if i + 1 in self.neck:
                    out_neck.append(fea)  # B, L, C
                if i + 1 in self.teachers:
                    B_f, L_f, C_f = fea.shape
                    # Infer grid size from patch embedding
                    grid_size = self.patch_embed.grid_size
                    H, W = grid_size
                    if L_f != H * W:
                        raise ValueError(f"Feature map length {L_f} does not match patch grid size {H}x{W}={H * W}.")
                    # Reshape features for teacher output: B, C, H, W
                    fea_t = fea.view(B_f, H, W, C_f).permute(0, 3, 1, 2).contiguous()
                    out_t.append(fea_t)

        return out_t, out_neck  # Return teacher features and neck features


# ========== ViT Decoder ==========
class ViT_Decoder(VisionTransformer):
	def __init__(self, students, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.students = students
		num_patches = self.patch_embed.num_patches
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
		trunc_normal_(self.pos_embed, std=.02)

	def forward(self, x):
		# taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
		# B, C, H, W = x.shape
		# x = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # B, L, C
		x = x + self.pos_embed
		x = self.pos_drop(x)

		out = []
		for idx, blk in enumerate(self.blocks):
			x = blk(x)
			if (idx + 1) in self.students:
				fea = x
				B, L, C = fea.shape
				H = int(np.sqrt(L))
				fea = fea.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
				out.append(fea)

		return [out[int(len(out)-1-i)] for i in range(len(out))]


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    # (Keep implementation as before, it uses kwargs directly)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    if 'flexi' in variant:
        try:
            _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=True)
        except TypeError:
            print("Warning: timm.models.layers.checkpoint_filter_fn might not support 'antialias'. Using default.")
            _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear')  # Fallback
    else:
        _filter_fn = checkpoint_filter_fn

    model = None
    attempts = 3
    for attempt in range(attempts):
        try:
            # Set num_classes=0 for encoders unless specified otherwise
            kwargs.setdefault('num_classes', 0)
            if kwargs['num_classes'] > 0:
                print(
                    f"Warning: Creating ViT_Encoder {variant} with num_classes={kwargs['num_classes']}. Ensure head is needed.")

            model = build_model_with_cfg(
                ViT_Encoder,  # Use our modified ViT_Encoder class
                variant,
                pretrained,
                pretrained_filter_fn=_filter_fn,
                **kwargs  # Pass remaining kwargs (like teachers, neck, img_size etc.)
            )
            break  # Success
        except Exception as e:
            print(f'Attempt {attempt + 1}/{attempts}: Failed to load model {variant}. Error: {e}')
            if attempt < attempts - 1:
                print('Retrying load model for ViTAD...')
                time.sleep(2)
            else:
                print(f'Failed to load model {variant} after {attempts} attempts.')
                raise e  # Re-raise the exception if all attempts fail

    if model is None:
        raise RuntimeError(f"Could not create model {variant} after {attempts} attempts.")

    return model



# --- Decoders (using ViT_Decoder class) ---
def _create_decoder(base_kwargs, **kwargs):
    # Helper to create ViT_Decoder instance
    model_kwargs = base_kwargs.copy()
    depth = kwargs.pop('depth')
    img_size = kwargs.get('img_size', 256)
    patch_size = model_kwargs.get('patch_size', 16)
    grid_size = (img_size // patch_size, img_size // patch_size)

    model_kwargs.update(kwargs)
    model_kwargs['depth'] = depth

    return ViT_Decoder(**model_kwargs)



# ========== Model Creator Mapping ==========
MODEL_CREATORS = {
    # Fusion
    'fusion': fusion,
    # Encoders
    # 'vit_small_patch16_224_1k': vit_small_patch16_224_1k,
    # 'vit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224_dino': vit_small_patch16_224_dino,
    # 'vit_small_patch8_224_dino': vit_small_patch8_224_dino,
    # 'vit_base_patch16_224_dino': vit_base_patch16_224_dino,
    # Decoders
    # 'de_vit_small_patch16_224_1k': de_vit_small_patch16_224_1k,
    # 'de_vit_base_patch16_224': de_vit_base_patch16_224,
    'de_vit_small_patch16_224_dino': de_vit_small_patch16_224_dino,
    # 'de_vit_base_patch16_224_dino': de_vit_base_patch16_224_dino,
    # 'de_vit_small_patch8_224_dino': de_vit_small_patch8_224_dino,
}


# ========== Helper for Model Creation and Loading ==========

def create_and_load_model(model_cfg: dict):
    """
    Creates a model using the corresponding function from MODEL_CREATORS
    and handles loading pretrained weights or checkpoints.

    Args:
        model_cfg (dict): A configuration dictionary with keys:
                          'name': Name of the model (str, key in MODEL_CREATORS).
                          'kwargs': Dictionary of keyword arguments for the model creator,
                                    including 'pretrained' (bool), 'checkpoint_path' (str),
                                    'strict' (bool), and model-specific args.

    Returns:
        The created (and potentially loaded) model instance.
    """
    model_name = model_cfg['name']
    kwargs = model_cfg['kwargs'].copy()  # Work on a copy

    if model_name not in MODEL_CREATORS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_CREATORS.keys())}")

    model_creator_fn = MODEL_CREATORS[model_name]

    # Extract loading-specific args from kwargs, providing defaults
    pretrained = kwargs.pop('pretrained', False)
    checkpoint_path = kwargs.pop('checkpoint_path', '')
    strict = kwargs.pop('strict', True)  # Default to strict loading for checkpoints

    # --- Model Instantiation ---
    # Pass the 'pretrained' flag and remaining specific kwargs to the creator
    print(f"Creating model: {model_name} with pretrained={pretrained}")
    print(f"Remaining kwargs for creator: {kwargs}")
    model = model_creator_fn(pretrained=pretrained, **kwargs)


    if checkpoint_path:
        print(f"Loading checkpoint for {model_name} from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu') # Load on CPU first to prevent GPU memory spike
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('net', ckpt.get('state_dict', ckpt.get('model', ckpt)))
                if state_dict is None:
                    print(
                        f"Warning: Could not find 'net', 'state_dict', or 'model' key in checkpoint for {model_name}.")
                    state_dict = ckpt  #
            else:
                state_dict = ckpt
                print(
                    f"Warning: Loaded checkpoint for {model_name} is not a dict. Assuming it's the state_dict directly.")

            if not isinstance(state_dict, dict):
                raise ValueError(
                    f"Loaded checkpoint for {model_name} is not a state dictionary (dict). Type: {type(state_dict)}")

            # --- State Dict Loading ---
            if isinstance(model, nn.Module):
                print(f"Loading state_dict into {model_name} with strict={strict}")
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
                    if missing_keys:
                        print(
                            f"Warning: Missing keys when loading checkpoint for {model_name} (strict={strict}): {missing_keys}")
                    if unexpected_keys:
                        print(
                            f"Warning: Unexpected keys when loading checkpoint for {model_name} (strict={strict}): {unexpected_keys}")
                except Exception as e:
                    # Attempt non-strict loading if strict loading failed
                    print(f"Error during strict={strict} state_dict loading for {model_name}: {e}")
                    if strict:  # Only try non-strict if strict loading was attempted and failed
                        print("Attempting load with strict=False...")
                        try:
                            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                            print(f"Successfully loaded state_dict for {model_name} with strict=False.")
                            if missing_keys: print(f"  Missing keys (strict=False): {missing_keys}")
                            if unexpected_keys: print(f"  Unexpected keys (strict=False): {unexpected_keys}")
                        except Exception as e2:
                            print(f"Failed to load state_dict for {model_name} even with strict=False. Error: {e2}")

            else:
                print(
                    f"Warning: Model {model_name} is not an nn.Module instance. Checkpoint loading might need custom handling.")

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint for {model_name} from {checkpoint_path}: {e}")

    return model


# ========== ViTAD Main Model ==========
class ViTAD(nn.Module):
    def __init__(self, model_t_cfg: dict, model_f_cfg: dict, model_s_cfg: dict):
        """
        Initializes the ViTAD model by creating its components directly.

        Args:
            model_t_cfg (dict): Configuration dictionary for the teacher model.
            model_f_cfg (dict): Configuration dictionary for the fusion model.
            model_s_cfg (dict): Configuration dictionary for the student model.
        """
        super(ViTAD, self).__init__()

        print("--- Initializing ViTAD ---")
        print("Teacher Config:", model_t_cfg)
        print("Fusion Config:", model_f_cfg)
        print("Student Config:", model_s_cfg)

        # Create sub-models using the helper function
        self.net_t = create_and_load_model(model_t_cfg)
        self.net_fusion = create_and_load_model(model_f_cfg)
        self.net_s = create_and_load_model(model_s_cfg)

        # Layer freezing logic
        self.frozen_layers = ['net_t']  # Freeze the teacher network by default

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        # print(f"Setting ViTAD to {'train' if mode else 'eval'} mode.")
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                # Ensure frozen layers are always in eval mode and grads are off
                self.freeze_layer(module)
                # print(f" > Layer '{mname}' is frozen (eval mode, no grads).")
            else:
                # Set other layers to the desired train/eval mode
                module.train(mode)
                # print(f" > Layer '{mname}' set to {'train' if mode else 'eval'} mode.")
        return self

    def forward(self, imgs):
        # Teacher forward pass (frozen, no gradients)
        self.net_t.eval()  # Ensure teacher is in eval mode
        with torch.no_grad():
            feats_t, feats_n = self.net_t(imgs)

        feats_n_detached = [f.detach() for f in feats_n]

        fused_features = self.net_fusion(feats_n_detached)
        feats_s = self.net_s(fused_features)

        return feats_t, feats_s

def load_default_model(device=None):
    image_size = 256
    teacher_embed_dim = 384
    teacher_patch_size = 16
    grid_size = image_size // teacher_patch_size
    if device is None:
        device = 'cpu'

    model_t_cfg = {
        'name': 'vit_small_patch16_224_dino',
        'kwargs': dict(
            pretrained=True,
            checkpoint_path='',
            strict=True,
            img_size=image_size,
            teachers=[3, 6, 9],
            neck=[12],
            num_classes=0
        )
    }

    # Fusion Network Config
    model_f_cfg = {
        'name': 'fusion',
        'kwargs': dict(
            pretrained=False,
            checkpoint_path='',
            strict=False,
            dim=teacher_embed_dim,  # Output dim of fusion matches teacher embed dim
            mul=len(model_t_cfg['kwargs']['neck'])  # Number of neck features to fuse
        )
    }

    model_s_cfg = {
        'name': 'de_vit_small_patch16_224_dino',  # Matches teacher base arch
        'kwargs': dict(
            pretrained=False,
            checkpoint_path='',
            strict=False,
            img_size=image_size,
            students=[3, 6, 9],
            depth=9,  # Depth of the decoder part
        )
    }

    try:
        net = ViTAD(
            model_t_cfg=model_t_cfg,
            model_f_cfg=model_f_cfg,
            model_s_cfg=model_s_cfg
        ).to(device)
        print(f"ViTAD model instantiated successfully. {device}")
    except Exception as e:
        print(f"\n !!! Error during ViTAD instantiation: {e} !!!")
        import traceback

        traceback.print_exc()
        exit()  # Stop if model creation fails

    return net
