"""
temp_data_processing.py
## TIMESTAMP @ 2025-04-07
## author: phuocddat
Employ MONAI transform component. <Temporary solution>
"""
import torch
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
from tqdm import tqdm
import matplotlib.pyplot as plt

from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRangePercentilesd, SpatialPadd, EnsureTyped, ConcatItemsd, MapTransform
)

SLICE_AXIS = 2
LOWER_SLICE_EXCLUSION = 80 # paper suggestion.
UPPER_SLICE_EXCLUSION = 26 # paper suggestion.
TARGET_SIZE = (256, 256) # or 64x64
IMAGE_KEYS = ['t1', 't1ce', 't2', 'flair']
LABEL_KEY = 'seg'
ALL_KEYS = IMAGE_KEYS + [LABEL_KEY]

"""
ExtractSliceAndLabeld(MapTransform)
"""
class ExtractSliceAndLabeld(MapTransform):
    def __init__(self, keys, slice_index_key="slice_index",
                 output_image_key="image", label_key="label",
                 image_level_label_key="image_level_label",
                 slice_axis=SLICE_AXIS):
        """
        Init params
        :param keys:
        :param slice_index_key:
        :param output_image_key:
        :param label_key:
        :param image_level_label_key:
        :param slice_axis:
        """
        super().__init__(keys, allow_missing_keys=False)
        self.slice_index_key = slice_index_key
        self.output_image_key = output_image_key
        self.label_key = label_key
        self.image_level_label_key = image_level_label_key
        self.slice_axis = slice_axis

    def __call__(self, data):
        d = dict(data) # Make a copy to modify
        slice_idx = d[self.slice_index_key]

        image_slices = []
        label_slice = None

        channel_dim = 0
        # The spatial slice axis index needs to account for the channel dim
        spatial_slice_axis_in_tensor = self.slice_axis + 1

        # Extract slice for each specified key
        for key in self.keys:
            # Check if the key corresponds to tensor data loaded by LoadImaged
            if isinstance(d[key], torch.Tensor) and d[key].ndim == 4:  # Expecting (C=1, H, W, D)
                img_3d = d[key]

                # Validate slice index against the dimension size
                if slice_idx >= img_3d.shape[spatial_slice_axis_in_tensor]:
                    print(
                        f"Error in transform: slice_idx {slice_idx} out of bounds for key '{key}' shape {img_3d.shape} axis {spatial_slice_axis_in_tensor}")

                    return d

                # Select slice along the spatial axis, keeping H, W dims
                # .select() reduces the dimension, so result is (C=1, H, W) if axis=3, or (C=1, W, D) if axis=1 etc.
                # We want the spatial slice (H, W), so remove channel dim first if needed
                img_slice_2d = img_3d.squeeze(channel_dim).select(dim=self.slice_axis, index=slice_idx)  # Shape: (H, W)

                if key == self.label_key:
                    label_slice = img_slice_2d
                elif key in IMAGE_KEYS:
                    image_slices.append(img_slice_2d)
        if len(image_slices) != len(IMAGE_KEYS):
            print(
                f"Warning: Expected {len(IMAGE_KEYS)} image slices for key '{self.output_image_key}', got {len(image_slices)}")

        if image_slices:
            d[self.output_image_key] = torch.stack(image_slices, dim=0)  # Shape: (4, H, W)
        else:
            d[self.output_image_key] = torch.empty((0, *TARGET_SIZE),
                                                   device=d[self.slice_index_key].device if isinstance(
                                                       d[self.slice_index_key], torch.Tensor) else None)  # Empty tensor

        if label_slice is not None:
            d[self.label_key] = label_slice  # Store the single-channel 2D label slice (H, W)
            is_diseased = 1 if torch.max(label_slice) > 0 else 0
            d[self.image_level_label_key] = torch.tensor(is_diseased, dtype=torch.long, device=label_slice.device)
        else:
            d[self.label_key] = torch.empty(TARGET_SIZE,
                                            device=d[self.slice_index_key].device if isinstance(d[self.slice_index_key],
                                                                                                torch.Tensor) else None)  # Placeholder
            d[self.image_level_label_key] = torch.tensor(-1, dtype=torch.long,
                                                         device=d[self.label_key].device)  # Indicate unknown

            # Clean up large 3D tensors that are no longer needed after slicing
        keys_to_remove = [k for k, v in d.items() if isinstance(v, torch.Tensor) and v.ndim == 4 and k in ALL_KEYS]
        for k in keys_to_remove:
            del d[k]

        return d


class MONAIBraTS2DSliceDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = os.path.expanduser(directory)
        self.data_list = self._create_data_list()
        # Store the original list separately if needed for raw loading
        self.raw_data_info = self.data_list[:]
        super().__init__(data=self.data_list, transform=transform)

    def _create_data_list(self):
        """Scans the directory and creates a list of dictionaries,
           each representing a single valid slice.
           :return: """
        data_list = []
        print(f"Scanning directory: {self.directory}")
        patient_folders = [d for d in os.listdir(self.directory)
                           if os.path.isdir(os.path.join(self.directory, d))]

        print(f"Found {len(patient_folders)} potential patient folders. Identifying valid slices...")
        for patient_id in tqdm(patient_folders, desc="Processing Patients"):
            patient_dir = os.path.join(self.directory, patient_id)
            files = os.listdir(patient_dir)
            # Store paths AND patient_id in the main dict for the transform pipeline
            file_paths_dict = {"patient_id": patient_id}
            found_modalities = set()

            # Collect file paths for this patient
            for f in files:
                if f.endswith('.nii.gz'):
                    parts = f.replace('.nii.gz', '').split('_')
                    if len(parts) > 1:
                        seqtype = parts[-1]
                        if seqtype in ALL_KEYS:
                            file_paths_dict[seqtype] = os.path.join(patient_dir, f)
                            found_modalities.add(seqtype)

            # Check if all required modalities and label are present
            required_set = set(ALL_KEYS)
            if found_modalities != required_set:
                continue

            try:
                seg_nii = nibabel.load(file_paths_dict[LABEL_KEY])
                num_slices = seg_nii.shape[-1]
                min_slice_idx = LOWER_SLICE_EXCLUSION
                max_slice_idx = num_slices - 1 - UPPER_SLICE_EXCLUSION

                if min_slice_idx > max_slice_idx:
                     continue

                for slice_idx in range(min_slice_idx, max_slice_idx + 1):
                    slice_info = file_paths_dict.copy()
                    slice_info["slice_index"] = slice_idx
                    data_list.append(slice_info)
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                continue

        print(f"Initialization complete. Found {len(data_list)} valid slices.")
        return data_list



# --- Visualization Functions ---

def _load_raw_slice(filepath, slice_idx, slice_axis=SLICE_AXIS):
    """Loads a single slice from a NIfTI file without processing."""
    try:
        nii_img = nibabel.load(filepath)
        img_data = nii_img.dataobj # Use dataobj for efficient slicing

        # Determine the number of spatial dimensions (usually 3 for BraTS)

        if img_data.ndim < 3:
             print(f"Warning: Image data at {filepath} has less than 3 dimensions ({img_data.ndim}). Cannot extract slice reliably.")
             return np.zeros((10, 10)) # Return dummy data

        # Create a slicer tuple dynamically
        slicer = [slice(None)] * img_data.ndim

        if slice_idx >= img_data.shape[slice_axis]:
             print(f"Warning: slice_idx {slice_idx} out of bounds for axis {slice_axis} (shape: {img_data.shape}) in {filepath}")
             # Attempt to load middle slice instead as fallback? Or return None/zeros?
             # Let's try the middle slice for visualization purposes
             slice_idx = img_data.shape[slice_axis] // 2
             print(f"Using middle slice index {slice_idx} instead.")


        slicer[slice_axis] = slice_idx # Set the specific slice index, for now.

        # Apply the slicer using tuple indexing
        slice_data = np.asarray(img_data[tuple(slicer)]) # Convert data proxy to numpy array

        return slice_data.astype(np.float32)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        # Print more specific error
        import traceback
        print(f"Error loading raw slice from {filepath} (index {slice_idx}, axis {slice_axis}):")
        traceback.print_exc() # Print full traceback
        return None

def visualize_monai_sample(dataset, index):
    """
    Visualizes raw modalities, processed modalities, labels, and metadata
    for a given sample index from the MONAI dataset.

    Args:
        dataset (MONAIBraTS2DSliceDataset): The instantiated dataset.
        index (int): The index of the sample to visualize.
    """
    if index >= len(dataset):
        print(f"Error: Index {index} is out of bounds for dataset size {len(dataset)}")
        return

    # Get the processed sample dictionary using the dataset's __getitem__
    processed_sample = dataset[index]

    # Access the original list stored BEFORE transforms were applied
    # This dictionary contains the file paths and original slice index
    raw_info = dataset.raw_data_info[index]
    patient_id = raw_info.get("patient_id", "N/A")
    slice_idx = raw_info.get("slice_index", "N/A")

    print(f"\n--- Visualizing Sample ---")
    print(f"Patient ID: {patient_id}, Original Slice Index: {slice_idx}")
    print(f"Processed sample keys: {processed_sample.keys()}")

    # More rows needed: Raw Modalities, Processed Modalities, Labels+Overlay+Info
    fig, axes = plt.subplots(3, 4, figsize=(16, 12)) # 3 rows, 4 columns
    axes = axes.ravel() # Flatten axes array for easy indexing

    # --- Plot Raw Modality Slices ---
    for i, key in enumerate(IMAGE_KEYS):
        ax = axes[i]
        raw_path = raw_info.get(key)
        if raw_path:
            raw_slice = _load_raw_slice(raw_path, slice_idx)
            if raw_slice is not None:
                im = ax.imshow(np.rot90(raw_slice), cmap='gray') # Rotate for better view
                ax.set_title(f"Raw {key.upper()} Slice")

            else:
                ax.set_title(f"Raw {key.upper()} (Error loading)")
        else:
            ax.set_title(f"Raw {key.upper()} (Path missing)")
        ax.axis('off')

    # --- Plot Processed Modality Slices (Channels of 'image') ---
    processed_image = processed_sample.get("image") # Shape (4, H, W)
    if processed_image is not None and processed_image.numel() > 0:
        processed_image_np = processed_image.cpu().numpy()
        for i in range(processed_image_np.shape[0]): # Iterate through channels
            ax = axes[i + 4] # Start from the 5th subplot (index 4)
            im = ax.imshow(np.rot90(processed_image_np[i]), cmap='gray')
            ax.set_title(f"Processed Ch {i} ({IMAGE_KEYS[i].upper()})")
            # fig.colorbar(im, ax=ax, shrink=0.8) # Optional colorbar
            ax.axis('off')
    else:
        for i in range(4):
             axes[i+4].set_title(f"Processed Ch {i} (Not found)")
             axes[i+4].axis('off')


    # --- Plot Labels and Overlay ---

    # Raw Label
    ax_raw_label = axes[8]
    raw_label_path = raw_info.get(LABEL_KEY)
    raw_label_slice = None
    if raw_label_path:
        raw_label_slice = _load_raw_slice(raw_label_path, slice_idx)
        if raw_label_slice is not None:
            ax_raw_label.imshow(np.rot90(raw_label_slice), cmap='gist_ncar', interpolation='nearest')
            ax_raw_label.set_title("Raw Label Slice")
        else:
            ax_raw_label.set_title("Raw Label (Error loading)")
    else:
        ax_raw_label.set_title("Raw Label (Path missing)")
    ax_raw_label.axis('off')

    # Processed Label
    ax_proc_label = axes[9]
    processed_label = processed_sample.get("label") # Shape (H, W)
    processed_label_np = None
    if processed_label is not None and processed_label.numel() > 0:
        processed_label_np = processed_label.cpu().numpy()
        im = ax_proc_label.imshow(np.rot90(processed_label_np), cmap='gist_ncar', interpolation='nearest')
        ax_proc_label.set_title("Processed Label Slice")
        # fig.colorbar(im, ax=ax_proc_label, shrink=0.8)
    else:
        ax_proc_label.set_title("Processed Label (Not found)")
    ax_proc_label.axis('off')


    # Overlay on Processed FLAIR (Channel 3)
    ax_overlay = axes[10]
    if processed_image is not None and processed_image.numel() > 0 and processed_label_np is not None:
        base_image = processed_image_np[3] # Use FLAIR channel (index 3)
        ax_overlay.imshow(np.rot90(base_image), cmap='gray')

        # Create a masked array for the label overlay - only show non-zero labels
        masked_label = np.ma.masked_where(processed_label_np == 0, processed_label_np)
        # Use a distinct colormap for the overlay (e.g., Reds, hot)
        overlay_cmap = plt.cm.get_cmap('Reds').copy()
        overlay_cmap.set_bad(alpha=0) # Make masked values transparent

        ax_overlay.imshow(np.rot90(masked_label), cmap=overlay_cmap, alpha=0.5, interpolation='nearest') # Transparency
        ax_overlay.set_title("Overlay on Processed FLAIR")
    else:
        ax_overlay.set_title("Overlay (Data missing)")
    ax_overlay.axis('off')

    # --- Display Metadata ---
    ax_info = axes[11]
    ax_info.axis('off')
    img_level_label_val = processed_sample.get("image_level_label", torch.tensor(-1)).item()
    label_text = f"Image Level Label: {img_level_label_val} "
    label_text += "(Diseased)" if img_level_label_val == 1 else "(Healthy)" if img_level_label_val == 0 else "(Unknown)"

    info_text = (
        f"Patient ID: {patient_id}\n"
        f"Slice Index (Original): {slice_idx}\n"
        f"{label_text}\n"
        f"Processed Img Shape: {processed_image.shape if processed_image is not None else 'N/A'}\n"
        f"Processed Lbl Shape: {processed_label.shape if processed_label is not None else 'N/A'}"
    )
    ax_info.text(0.05, 0.95, info_text, ha='left', va='top', fontsize=10, wrap=True)
    ax_info.set_title("Metadata")


    # --- Final Touches ---
    fig.suptitle(f"BraTS Sample Visualization - Patient: {patient_id}, Slice: {slice_idx}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


# ---  ---
# 1. Set the path
brats_data_path = '/mnt/Data/12.Data/Medical_Data/BraTS2021/BraTS2021_Training_Data/'

# 2. Set up compose
train_transforms = Compose(
    [
        LoadImaged(keys=ALL_KEYS, image_only=False, ensure_channel_first=True), # loads C,H,W,D
        Orientationd(keys=ALL_KEYS, axcodes="RAS"),
        ScaleIntensityRangePercentilesd(
            keys=IMAGE_KEYS, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        # ExtractSlice needs access to paths and slice_index, LoadImaged keeps them
        ExtractSliceAndLabeld(
            keys=ALL_KEYS, # Will process these keys if they are tensors
            slice_index_key="slice_index",
            output_image_key="image",
            label_key=LABEL_KEY,
            image_level_label_key="image_level_label"
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=TARGET_SIZE,
            mode="constant",
            allow_missing_keys=True
        ),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.int64, allow_missing_keys=True),
    ]
)


# 3. Create the dataset instance

dataset = MONAIBraTS2DSliceDataset(brats_data_path, transform=train_transforms)

# 3. Visualize a specific sample (e.g., the 10th valid slice found)
if len(dataset) > 10:
    visualize_monai_sample(dataset, 10)
elif len(dataset) > 0:
     visualize_monai_sample(dataset, 0) # Visualize the first one if dataset is small
else:
    print("Dataset is empty, cannot visualize.")
