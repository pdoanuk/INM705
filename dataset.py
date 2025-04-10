import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import numpy as np
## TIMESTAMP @ 2025-04-10T23:45:47
## author: phuocddat
## start
# very basic pipeline to work
## end --
class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, is_train=True, resize=256, cropsize=224):
        """

        :param root_dir:
        :param category:
        :param is_train:
        :param resize:
        :param cropsize:
        """
        self.root_dir = root_dir
        self.category = category
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.LANCZOS),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet normalization.
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize(resize, Image.NEAREST), # Use NEAREST for masks
            transforms.CenterCrop(cropsize),
            transforms.ToTensor()
        ])

        if is_train:
            self.image_paths = sorted(glob.glob(os.path.join(root_dir, category, 'train', 'good', '*.png')))
        else:
            self.image_paths = sorted(glob.glob(os.path.join(root_dir, category, 'test', '*', '*.png')))
            self.gt_paths = sorted(glob.glob(os.path.join(root_dir, category, 'ground_truth', '*', '*.png')))
            # Ensure 'good' images have a placeholder or black mask
            self.gt_paths_map = {os.path.basename(os.path.dirname(p)) + '_' + os.path.basename(p).split('.')[0]: p for p in self.gt_paths}


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.is_train:
            return img, torch.zeros(1) # Return image and label 0 (normal)
        else:
            # Determine if the image is anomalous based on its path # Maybe for now.
            parts = img_path.split(os.sep)
            defect_type = parts[-2]
            is_normal = (defect_type == 'good')
            label = 0 if is_normal else 1

            # Load ground truth mask
            if is_normal:
                gt_mask = torch.zeros([1, self.cropsize, self.cropsize]) # Black mask for normal images
            else:
                key = defect_type + '_' + os.path.basename(img_path).split('.')[0]
                gt_path = self.gt_paths_map.get(key)
                if gt_path:
                    gt_mask = Image.open(gt_path).convert('L') # Ensure grayscale
                    gt_mask = self.gt_transform(gt_mask)
                    gt_mask = torch.where(gt_mask > 0.5, torch.tensor(1.0), torch.tensor(0.0)) # Binarize
                else: # Should not happen with standard MVTec structure, but good practice
                     print(f"Warning: Ground truth mask not found for {img_path}")
                     gt_mask = torch.zeros([1, self.cropsize, self.cropsize])


            return img, gt_mask, torch.tensor(label) # Return image, mask, label (0 or 1)