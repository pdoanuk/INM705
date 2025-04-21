import torch
import torch.nn as nn
import timm
from config import args
## TIMESTAMP @ 2025-04-10T23:45:47
## author: phuocddat
## start
# very basic pipeline to work
## end -

class VitDecoderExp(nn.Module):
    def __init__(self, model_name='vit_base_patch16_384', pretrained=True):
        super(VitDecoderExp, self).__init__()
        ## self.args = args
        self.image_size = args.image_size
        self.patch_size = args.patch_size
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"Image size ({self.image_size}) must be divisible by patch size ({self.patch_size})")

        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        # Remove the classification head
        self.encoder.head = nn.Identity()

        embed_dim = self.encoder.embed_dim
        num_patches = self.encoder.patch_embed.num_patches
        self.num_patches_side = int(num_patches ** 0.5)

        # Project ViT output back to a spatial feature map
        self.proj = nn.Conv2d(embed_dim, 768, kernel_size=1)  # Reduce dimensionality
        self.decoder = nn.Sequential(
            ## Block 1
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=3),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            ## Block 2
            nn.ConvTranspose2d(256, 128, kernel_size=3),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            ## Block 3
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True),

        )

        self.decoder_bano2d = nn.Sequential(
            ## Block 1
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=3),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            ## Block 2
            nn.ConvTranspose2d(256, 128, kernel_size=3),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            ## Block 3
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True),

        )

        self.upscale = nn.UpsamplingBilinear2d((self.image_size, self.image_size))
        self.tanh = nn.Tanh()

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        if input_h != self.image_size or input_w != self.image_size:
            raise ValueError(
                f"Input image size ({input_h}x{input_w}) does not match model's configured size ({self.image_size}x{self.image_size})")
        # x shape: (B, C, H, W) _ important to check
        # Encode
        features = self.encoder.forward_features(x)  # Get patch embeddings + cls token
        # print(features.shape)

        # Process features for decoder
        # Use only patch tokens
        patch_tokens = features[:, 1:, :]  # Exclude CLS token: (B, num_patches, embed_dim)
        B, N, E = patch_tokens.shape
        H = W = self.num_patches_side
        patch_tokens_spatial = patch_tokens.permute(0, 2, 1).reshape(B, E, H, W)
        # patch_tokens = patch_tokens.transpose(1, 2)
        # patch_tokens_spatial = patch_tokens.reshape(features.shape[0], -1, self.image_size // self.patch_size,
        #                   self.image_size // self.patch_size)
        # Project and Decode
        # projected_features = self.proj(patch_tokens_spatial)
        reconstructed_x = self.decoder(patch_tokens_spatial)
        reconstructed_x = self.upscale(reconstructed_x)
        reconstructed_x = self.tanh(reconstructed_x)

        return reconstructed_x



class ViTAutoencoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        # Remove the classification head
        self.encoder.head = nn.Identity()

        embed_dim = self.encoder.embed_dim
        num_patches = self.encoder.patch_embed.num_patches
        self.num_patches_side = int(num_patches ** 0.5)

        # Project ViT output back to a spatial feature map
        self.proj = nn.Conv2d(embed_dim, 256, kernel_size=1)  # Reduce dimensionality

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.Sigmoid()  # Output pixel values between 0 and 1
        )

        # Store normalization params for denormalization if needed for visualization/loss
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # x shape: (B, C, H, W) _ important to check
        # Encode
        features = self.encoder.forward_features(x)  # Get patch embeddings + cls token
        #print(features.shape)

        # Process features for decoder
        # Use only patch tokens
        patch_tokens = features[:, 1:, :]  # Exclude CLS token: (B, num_patches, embed_dim)

        B, N, E = patch_tokens.shape
        H = W = self.num_patches_side
        patch_tokens_spatial = patch_tokens.permute(0, 2, 1).reshape(B, E, H, W)

        # Project and Decode
        projected_features = self.proj(patch_tokens_spatial)
        reconstructed_x = self.decoder(projected_features)
        return reconstructed_x

    def denormalize(self, x):
        # x is the output of the sigmoid (0 to 1 range)
        # Should check carefully...

        if x.device != self.norm_mean.device:
            self.norm_mean = self.norm_mean.to(x.device)
            self.norm_std = self.norm_std.to(x.device)


        #  output `x` needs denormalizing from standard norm space
        denorm_x = x * self.norm_std + self.norm_mean
        return torch.clamp(denorm_x, 0., 1.)  # Clamp to valid image range


