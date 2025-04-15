## TIMESTAMP @ 2025-04-10T23:45:47
## author: phuocddat
## start
# very basic pipeline to work
## end --

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from config import CLASS_NAMES, mean_train, std_train
from pkg_resources import parse_version


if parse_version(Image.__version__)>=parse_version('10.0.0'):
    Image.ANTIALIAS=Image.LANCZOS


class MVTecDataset(Dataset):
    def __init__(self, args, dataset_path, class_name, is_train=True, resize=384, transform_x=transforms.ToTensor()):
        super(MVTecDataset, self).__init__()
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.args = args
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize + self.args.image_size // 12
        self.transform_x = transform_x

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_mask = transforms.Compose([transforms.Resize(self.args.image_size, Image.NEAREST),
                                         transforms.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if mask == None:
            mask = torch.zeros([1, self.args.image_size, self.args.image_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))

            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)




def get_dataloader(args):
    obj_mean = mean_train
    obj_std = std_train
    trainT = transforms.Compose(
        [
            transforms.Resize(args.image_size, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    validT = transforms.Compose(
        [
            transforms.Resize(args.image_size, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    testT = transforms.Compose(
        [
            transforms.Resize(args.image_size, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=obj_mean, std=obj_std),
        ]
    )

    train_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=True,
        resize=args.image_size,
        transform_x=trainT,
    )
    valid_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=True,
        resize=args.image_size,
        transform_x=validT,
    )
    img_nums = len(train_dataset)

    indices = list(range(img_nums))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    split = int(np.floor(args.val_ratio * img_nums))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    test_dataset = MVTecDataset(
        args,
        args.dataset_path,
        class_name=args.obj,
        is_train=False,
        resize=args.image_size,
        transform_x=testT,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, sampler=valid_sampler
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader