# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn
from functools import partial

# Import necessary components from timm and torch
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
    # pretrained flag is ignored for Fusion
    model = Fusion(**kwargs)
    return model


# ========== ViT Encoder Base Classes ==========
class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, teachers, neck, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teachers
        self.neck = neck
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        # Adjust pos_embed size for cls_token and dist_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        # Ensure head_dist uses self.num_classes defined in VisionTransformer.__init__
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # Use _init_weights from the parent class if applicable
        if hasattr(self, '_init_weights'):
            self.head_dist.apply(self._init_weights)
        else:  # Basic init if _init_weights not found
            nn.init.zeros_(self.head_dist.bias)
            nn.init.normal_(self.head_dist.weight, std=0.02)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        # Ensure pos_embed shape matches concatenated sequence length
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Sequence length after token concat ({x.shape[1]}) doesn't match pos_embed length ({self.pos_embed.shape[1]})")

        x = x + self.pos_embed
        x = self.pos_drop(x)

        out_t, out_neck = [], []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            # Features excluding cls and dist tokens
            fea = x[:, 2:]
            if (idx + 1) in self.neck:
                out_neck.append(fea)  # B, L, C
            if (idx + 1) in self.teachers:
                B_f, L_f, C_f = fea.shape
                H = W = int(np.sqrt(L_f))  # Assume square patch grid
                if H * W != L_f:
                    # Try to infer grid shape if possible (might need img_size and patch_size)
                    grid_size = self.patch_embed.grid_size
                    if L_f == grid_size[0] * grid_size[1]:
                        H, W = grid_size
                    else:
                        raise ValueError(
                            f"Feature map size {L_f} is not a perfect square and doesn't match grid_size {grid_size}. Cannot reshape.")
                # Reshape features for teacher output: B, C, H, W
                fea_t = fea.view(B_f, H, W, C_f).permute(0, 3, 1, 2).contiguous()
                out_t.append(fea_t)

        # Return teacher features and neck features
        return out_t, out_neck

    def forward(self, x):
        # Standard forward returns teacher and neck features for ViTAD
        out_t, out_neck = self.forward_features(x)
        return out_t, out_neck


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
            # Feature extraction logic needs careful review with checkpointing
            print("Warning: Feature extraction from intermediate blocks might be affected by grad_checkpointing.")
            # Simple sequential execution for feature extraction (might be incorrect with checkpointing)
            current_input = x
            for i, blk in enumerate(self.blocks):
                x = blk(current_input)
                # Feature extraction logic (same as else block, applied sequentially)
                fea = x[:, self.num_prefix_tokens:]
                if i + 1 in self.neck:
                    out_neck.append(fea)
                if i + 1 in self.teachers:
                    B_f, L_f, C_f = fea.shape
                    H = W = self.patch_embed.grid_size[0]  # Use grid size from patch_embed
                    if L_f != H * W: raise ValueError(f"Feature len {L_f} != grid H*W {H * W}")
                    fea_t = fea.view(B_f, H, W, C_f).permute(0, 3, 1, 2).contiguous()
                    out_t.append(fea_t)
                current_input = x  # Update input for next block in checkpoint sequence (conceptual)

        else:  # Standard sequential execution
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
# ========== Specific Model Creation Functions ==========

# Helper for creating timm-based ViT models
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
    depth = kwargs.pop('depth')  # Decoder depth must be specified
    # Infer decoder_seq_len if possible from img_size and patch_size
    # This assumes the decoder input features correspond to the full patch grid
    img_size = kwargs.get('img_size', 256)  # Assume default or get from kwargs
    patch_size = model_kwargs.get('patch_size', 16)  # Get from base_kwargs
    grid_size = (img_size // patch_size, img_size // patch_size)
    # decoder_seq_len = grid_size[0] * grid_size[1]
    # model_kwargs['decoder_seq_len'] = decoder_seq_len
    # print(f"Inferred decoder sequence length: {decoder_seq_len} (Grid: {grid_size})")

    model_kwargs.update(kwargs)  # Merge remaining kwargs (students, etc.)
    model_kwargs['depth'] = depth  # Set the specific decoder depth

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
    # The creator function itself handles 'pretrained' loading from standard sources
    model = model_creator_fn(pretrained=pretrained, **kwargs)

    # --- Custom Checkpoint Loading (if path provided) ---
    # This overrides any weights loaded by pretrained=True within the creator
    if checkpoint_path:
        print(f"Loading checkpoint for {model_name} from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')

            # Determine the state dict within the checkpoint file
            if isinstance(ckpt, dict):
                # Common keys for state dicts in checkpoints
                state_dict = ckpt.get('net', ckpt.get('state_dict', ckpt.get('model', ckpt)))
                if state_dict is None:
                    print(
                        f"Warning: Could not find 'net', 'state_dict', or 'model' key in checkpoint for {model_name}. Attempting to load entire checkpoint dict as state_dict.")
                    state_dict = ckpt  # Assume the whole dict is the state_dict
            else:
                # Assume ckpt *is* the state_dict itself (less common)
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
                # Handle cases where the model object might not be a standard nn.Module
                print(
                    f"Warning: Model {model_name} is not an nn.Module instance. Checkpoint loading might need custom handling.")

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint for {model_name} from {checkpoint_path}: {e}")
            # Decide if you want to raise the error or continue
            # raise e

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
        # print(f"Froze layer: {module.__class__.__name__}") # Optional: logging

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
            feats_t, feats_n = self.net_t(imgs)  # list of teacher features, list of neck features

        # Detach neck features before passing to trainable fusion/student parts
        # Although no_grad context helps, explicit detach adds safety if grads were needed up to neck
        feats_n_detached = [f.detach() for f in feats_n]

        # Fusion and Student forward pass (respects self.training mode)
        # Ensure fusion and student are in the correct mode set by self.train()
        fused_features = self.net_fusion(feats_n_detached)
        feats_s = self.net_s(fused_features)

        # Return teacher features (for loss) and student features (for loss)
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

# ========== Main Execution Block ==========
if __name__ == '__main__':
    # Note: fvcore is an external dependency, ensure it's installed (`pip install fvcore`)
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count

        fvcore_available = True
    except ImportError:
        print("fvcore not found. Skipping FLOPs and detailed parameter count.")
        print("Install with: pip install fvcore")
        fvcore_available = False

    # --- Configuration Setup ---
    bs = 2
    image_size = 256
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Ensure input tensor is on the correct device
    x = torch.randn(bs, 3, image_size, image_size, device=device)

    # Define configurations for each part of ViTAD using dictionaries
    # Teacher Network Config
    model_t_cfg = {
        'name': 'vit_small_patch16_224_dino',
        'kwargs': dict(
            pretrained=True,
            checkpoint_path='',  # '/path/to/teacher_checkpoint.pth' # Optional override
            strict=True,  # Strict loading for pretrained usually desired
            img_size=image_size,
            teachers=[3, 6, 9],  # Layers for teacher features
            neck=[12],  # Layer(s) for fusion input
            num_classes=0  # Ensure no classification head
        )
    }

    # Calculate expected fusion input dimension based on teacher neck output
    # Assuming teacher is ViT-Small (embed_dim=384) and neck has 1 layer
    teacher_embed_dim = 384  # Hardcoded based on 'vit_small...' - better to infer if possible
    fusion_input_dim = teacher_embed_dim * len(model_t_cfg['kwargs']['neck'])

    # Fusion Network Config
    model_f_cfg = {
        'name': 'fusion',
        'kwargs': dict(
            pretrained=False,
            checkpoint_path='',
            strict=False,  # Often False for intermediate/custom layers
            dim=teacher_embed_dim,  # Output dim of fusion matches teacher embed dim
            mul=len(model_t_cfg['kwargs']['neck'])  # Number of neck features to fuse
        )
    }
    # Verify fusion input dim matches expectation
    assert model_f_cfg['kwargs']['dim'] * model_f_cfg['kwargs']['mul'] == fusion_input_dim, \
        f"Fusion input dimension mismatch: Expected {fusion_input_dim}, Config implies {model_f_cfg['kwargs']['dim'] * model_f_cfg['kwargs']['mul']}"

    # Student Network (Decoder) Config
    # Calculate expected decoder sequence length
    teacher_patch_size = 16  # Hardcoded based on 'patch16' - infer if possible
    grid_size = image_size // teacher_patch_size
    decoder_seq_len = grid_size * grid_size

    model_s_cfg = {
        'name': 'de_vit_small_patch16_224_dino',  # Matches teacher base arch
        'kwargs': dict(
            pretrained=False,
            checkpoint_path='',  # '/path/to/student_checkpoint.pth'
            strict=False,  # Often False for decoders trained from scratch
            img_size=image_size,  # Needed for H, W inference in decoder
            students=[3, 6, 9],  # Decoder layers corresponding to teacher layers
            depth=9,  # Depth of the decoder part
        )
    }

    # --- Model Instantiation ---
    print(f"\nInstantiating ViTAD model on {device}...")
    # Pass the configuration dictionaries directly to ViTAD
    try:
        net = ViTAD(
            model_t_cfg=model_t_cfg,
            model_f_cfg=model_f_cfg,
            model_s_cfg=model_s_cfg
        ).to(device)
        print("ViTAD model instantiated successfully.")
    except Exception as e:
        print(f"\n !!! Error during ViTAD instantiation: {e} !!!")
        import traceback

        traceback.print_exc()
        exit()  # Stop if model creation fails

    # Set model to eval mode for analysis, respecting frozen layers
    net.eval()
    # Explicitly call train(False) to ensure freeze_layer logic runs correctly
    net.train(mode=False)

    print("\n--- Model Structure (Top Level) ---")
    print(net)
    # Optional: Print sub-module details
    print("\n--- Teacher Network ---")
    print(net.net_t)
    print("\n--- Fusion Network ---")
    print(net.net_fusion)
    print("\n--- Student Network ---")
    print(net.net_s)
    print("-----------------------------------\n")

    # --- Analysis ---
    print("Running analysis...")
    try:
        # Parameter Count (using our helper for trainable)
        params_trainable = get_net_params(net)
        total_params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"Parameters: {total_params:.3f} M (Total), {params_trainable:.3f} M (Trainable)")

        flops = -1.0  # Default value if fvcore fails
        if fvcore_available:
            print("Analyzing FLOPs with fvcore...")
            # Ensure model is in eval mode for FlopCountAnalysis
            net.eval()
            flops_analyzer = FlopCountAnalysis(net, x)
            print("FLOPs Analysis Table (fvcore):")
            # Show top-level blocks, adjust max_depth as needed
            print(flop_count_table(flops_analyzer, max_depth=4))
            flops = flops_analyzer.total() / bs / 1e9  # GFLOPs per image
            print(f"Total GFLOPs per image: {flops:.3f} G")
        else:
            print("Skipping FLOPs calculation (fvcore not available).")

        # Speed Test
        print("\nRunning speed test...")
        # Ensure model is in eval mode and correct device
        net.eval()
        net.to(device)
        x = x.to(device)  # Ensure input is on the right device

        with torch.no_grad():
            # Warm-up runs
            print("Warm-up...")
            warmup_runs = max(5, bs)  # Ensure enough runs for stable timing
            for _ in range(warmup_runs):
                _ = net(x)

            # Timed runs
            print("Timing...")
            t_s = get_timepc()
            num_timed_runs = max(10, bs * 2)  # More runs for better average
            for i in range(num_timed_runs):
                _ = net(x)
                # Optional: Print progress for long runs
                # if (i+1) % 10 == 0: print(f" Timing run {i+1}/{num_timed_runs}")
            t_e = get_timepc()

        elapsed_time = t_e - t_s
        total_images = bs * num_timed_runs
        speed = total_images / elapsed_time if elapsed_time > 0 else 0  # Images per second
        latency = elapsed_time / num_timed_runs * 1000  # ms per batch

        print(f"Elapsed time for {num_timed_runs} runs ({total_images} images): {elapsed_time:.3f} s")
        print(f"Speed: {speed:.3f} images/sec")
        print(f"Latency per batch (bs={bs}): {latency:.3f} ms")

        print('\n--- Summary ---')
        print(f"[ GFLOPs/img: {flops:>6.3f} G ]\t"
              f"[ Params: {total_params:>6.3f} M total / {params_trainable:>6.3f} M trainable ]\t"
              f"[ Speed: {speed:>7.3f} img/s ]")
        print('---------------\n')


    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")
        import traceback

        traceback.print_exc()
