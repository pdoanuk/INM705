## TIMESTAMP @ 2025-04-10T23:45:47
## author: phuocddat
## start
# very basic pipeline to work
## end --

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
from config import CLASS_NAMES, mean_train, std_train, args
from pkg_resources import parse_version

if parse_version(Image.__version__) >= parse_version('10.0.0'):
    Image.ANTIALIAS = Image.LANCZOS


class MVTecDataset(Dataset):
    def __init__(self, args, dataset_path, class_name, is_train=True, resize=384, transform_x=None):
        super(MVTecDataset, self).__init__()
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.args = args
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize + self.args.image_size // 12
        if transform_x is not None:
            self.transform_x = transform_x
        else:
            self.transform_x = transforms.Compose(
                [transforms.Resize(args.image_size, Image.ANTIALIAS),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=mean_train, std=std_train),
                 ])

        # load dataset
        self.x, self.y, self.mask, self.cls_name = self.load_dataset_folder()

        self.transform_mask = transforms.Compose([transforms.Resize(self.args.image_size, Image.NEAREST),
                                                  transforms.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask, cls_name = self.x[idx], self.y[idx], self.mask[idx], self.cls_name[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if mask is None:
            mask = torch.zeros([1, self.args.image_size, self.args.image_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        return x, y, mask, cls_name

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, cls_name = [], [], [], []

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
            cls_name.extend([self.class_name] * len(img_fpath_list))
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

        assert len(x) == len(y) == len(mask) == len(cls_name), \
            f'Data loading mismatch for {self.class_name}/{phase}: len(x)={len(x)}, len(y)={len(y)}, len(mask)={len(mask)}, len(cls_name)={len(cls_name)}'
        return list(x), list(y), list(mask), list(cls_name)


def get_dataloader(args):
    transform_func = transforms.Compose([transforms.Resize(args.image_size, Image.ANTIALIAS),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean_train, std=std_train),
                                         ])
    #
    # validT = transforms.Compose(
    #     [
    #         transforms.Resize(args.image_size, Image.ANTIALIAS),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=obj_mean, std=obj_std),
    #     ]
    # )
    #
    # testT = transforms.Compose(
    #     [
    #         transforms.Resize(args.image_size, Image.ANTIALIAS),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=obj_mean, std=obj_std),
    #     ]
    # )

    train_dataset = MVTecDataset(args,
                                 args.dataset_path,
                                 class_name=args.obj,
                                 is_train=True,
                                 resize=args.image_size,
                                 transform_x=transform_func
                                 )
    valid_dataset = MVTecDataset(args,
                                 args.dataset_path,
                                 class_name=args.obj,
                                 is_train=True,
                                 resize=args.image_size,
                                 transform_x=transform_func,
                                 )
    img_nums = len(train_dataset)
    indices = list(range(img_nums))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    split = int(np.floor(args.val_ratio * img_nums))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    test_dataset = MVTecDataset(args,
                                args.dataset_path,
                                class_name=args.obj,
                                is_train=False,
                                resize=args.image_size,
                                transform_x=transform_func,
                                )

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(valid_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, val_loader, test_loader

def get_loader_full(args, list_categories=None):
    val_ratio = args.val_ratio
    train_ratio = 1 - val_ratio
    print("Loading datasets for categories:", list_categories)
    total_train_size = 0
    total_val_size = 0
    total_test_size = 0

    transform_func = transforms.Compose([transforms.Resize(args.image_size, Image.ANTIALIAS),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean_train, std=std_train),
                                         ])
    if list_categories is None:
        list_categories = CLASS_NAMES

    train_datasets_list = []
    val_datasets_list = []
    test_datasets_list = []
    for category in list_categories:
        train_val_dataset = MVTecDataset(args,
                                     args.dataset_path,
                                     class_name=category,
                                     is_train=True,
                                     resize=args.image_size,
                                     transform_x=transform_func
                                     )
        train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_ratio, val_ratio])
        train_datasets_list.append(train_dataset)
        val_datasets_list.append(val_dataset)

        test_dataset = MVTecDataset(args,
                                     args.dataset_path,
                                     class_name=category,
                                     is_train=True,
                                     resize=args.image_size,
                                     transform_x=transform_func
                                     )
        test_datasets_list.append(test_dataset)

        # Print sizes for verification
        cat_train_size = len(train_dataset)
        cat_val_size = len(val_dataset)
        cat_test_size = len(test_dataset)
        print(f"  Category: {category}")
        print(f"    Train subset size: {cat_train_size}")
        print(f"    Val subset size:   {cat_val_size}")
        print(f"    Test dataset size: {cat_test_size}")
        total_train_size += cat_train_size
        total_val_size += cat_val_size
        total_test_size += cat_test_size

    print("-" * 30)
    print(f"Total Train samples across categories: {total_train_size}")
    print(f"Total Val samples across categories:   {total_val_size}")
    print(f"Total Test samples across categories:  {total_test_size}")
    print("-" * 30)

    train_dataset_full = ConcatDataset(train_datasets_list)
    val_dataset_full = ConcatDataset(val_datasets_list)

    train_loader_full = DataLoader(train_dataset_full,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True,
                                   drop_last=True)
    val_loader_full = DataLoader(val_dataset_full,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True,
                                   drop_last=True)
    test_dataset_full = ConcatDataset(test_datasets_list)
    test_loader_full = DataLoader(test_dataset_full,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)

    return train_loader_full, val_loader_full, test_loader_full


# --- Example Usage ---
if __name__ == '__main__':

    # Make sure the dataset path in `args` is correct
    # Example: Download MVTec AD to './mvtec_anomaly_detection'
    if not os.path.exists(args.dataset_path):
        print(f"Dataset path {args.dataset_path} not found.")
        print("Please download the MVTec AD dataset and update the path in the script.")
        # Example download command (adjust as needed):
        # wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
        # tar -xf mvtec_anomaly_detection.tar.xz
    else:
        print(f"Using MVTec dataset from: {args.dataset_path}")

        # --- Test get_loader (single category) ---
        print("\n--- Testing get_loader (single category: 'bottle') ---")
        try:
            args.obj = 'bottle'
            train_loader_bottle, val_loader_bottle, test_loader_bottle = get_dataloader(args)

            # Fetch a batch to test
            print("Fetching one batch from bottle train loader...")
            x_batch, y_batch, mask_batch, cls_batch = next(iter(train_loader_bottle))
            print("Train Batch Shapes:")
            print("  Images:", x_batch.shape, x_batch.dtype)
            print("  Labels:", y_batch.shape, y_batch.dtype)
            print("  Masks:", mask_batch.shape, mask_batch.dtype)
            print("  Class Names:", cls_batch) # List of strings

            print("\nFetching one batch from bottle test loader...")
            x_batch_test, y_batch_test, mask_batch_test, cls_batch_test = next(iter(test_loader_bottle))
            print("Test Batch Shapes:")
            print("  Images:", x_batch_test.shape)
            print("  Labels:", y_batch_test.shape)
            print("  Masks:", mask_batch_test.shape)
            print("  Class Names:", cls_batch_test)

        except FileNotFoundError as e:
            print(f"Error loading single category: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during get_loader test: {e}")


        # --- Test get_loader_full (all categories) ---
        print("\n--- Testing get_loader_full (all categories) ---")
        try:
            train_loader_full, val_loader_full, test_loader_full = get_loader_full(args)

            print("Fetching one batch from full train loader...")
            x_batch_full, y_batch_full, mask_batch_full, cls_batch_full = next(iter(train_loader_full))
            print("Full Train Batch Shapes:")
            print("  Images:", x_batch_full.shape, x_batch_full.dtype)
            print("  Labels:", y_batch_full.shape, y_batch_full.dtype)
            print("  Masks:", mask_batch_full.shape, mask_batch_full.dtype)
            print("  Class Names:", cls_batch_full) # Should contain names from different classes potentially

            print("\nFetching one batch from full test loader...")
            x_batch_full_test, y_batch_full_test, mask_batch_full_test, cls_batch_full_test = next(iter(test_loader_full))
            print("Full Test Batch Shapes:")
            print("  Images:", x_batch_full_test.shape)
            print("  Labels:", y_batch_full_test.shape)
            print("  Masks:", mask_batch_full_test.shape)
            print("  Class Names:", cls_batch_full_test)

        except FileNotFoundError as e:
            print(f"Error loading full dataset: {e}")
            print("Check if all category subdirectories exist in the dataset path.")
        except Exception as e:
             print(f"An unexpected error occurred during get_loader_full test: {e}")


        # --- Test get_loader_full (specific categories) ---
        print("\n--- Testing get_loader_full (specific categories: ['carpet', 'grid']) ---")
        try:
            specific_categories = ['carpet', 'grid']
            train_loader_spec, val_loader_spec, test_loader_spec = get_loader_full(args, list_categories=specific_categories)

            print("Fetching one batch from specific train loader...")
            x_batch_spec, y_batch_spec, mask_batch_spec, cls_batch_spec = next(iter(train_loader_spec))
            print("Specific Train Batch Shapes:")
            print("  Images:", x_batch_spec.shape, x_batch_spec.dtype)
            print("  Labels:", y_batch_spec.shape, y_batch_spec.dtype)
            print("  Masks:", mask_batch_spec.shape, mask_batch_spec.dtype)
            print("  Class Names:", cls_batch_spec) # Should only contain 'carpet' or 'grid'

        except FileNotFoundError as e:
            print(f"Error loading specific categories: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during get_loader_full specific test: {e}")