import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import random
from pathlib import Path
from PIL import Image
import os
import sys
from typing import Optional, Sequence

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Syntract'))

from Syntract.cumulative import process_patches_inmemory
import albumentations as A


def extract_foreground_coords(label: np.ndarray, min_area=0):
    """Extract all foreground pixel coordinates."""
    coords = np.argwhere(label > 0)  # Assumes 0 = background
    if coords.shape[0] == 0:
        return []
    return [tuple(coord) for coord in coords]

class HistFinetuneDatasetCachedForeground(Dataset):
    def __init__(self, slice_samples, patch_h=1024, patch_w=1024,
                 num_random_patches=10, transform=None, foreground_oversample_ratio=0.5): #0.5 or 0.33
        self.slice_samples = slice_samples
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_random_patches = num_random_patches
        self.transform = transform
        self.foreground_oversample_ratio = foreground_oversample_ratio

        # Use the exact statistics from nnUNet's plans.json
        self.channel_means = [
            26.370181406062926,  # channel 0
            28.796625947529606,  # channel 1
            25.990918044351993   # channel 2
        ]
        self.channel_stds = [
            31.776848516250062,  # channel 0
            28.548426642727726,  # channel 1
            16.703267170423146   # channel 2
        ]
        
        print(f"Using nnUNet statistics:")
        print(f"Means: {self.channel_means}")
        print(f"Stds: {self.channel_stds}")

        self.flat_samples = []
        for sample in slice_samples:
            # Load the label image to extract foreground locations
            label = np.array(Image.open(sample['label_path']).convert("L")).astype(np.int64)
            fg_coords = extract_foreground_coords(label)

            # Add to metadata
            sample['fg_coords'] = fg_coords
            sample['label_shape'] = label.shape  # used for fallback cropping

            for _ in range(num_random_patches):
                self.flat_samples.append(sample)

    def normalize_image(self, image):
        """
        Apply Z-score normalization using nnUNet's dataset statistics
        """
        normalized = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            normalized[:, :, c] = (image[:, :, c] - self.channel_means[c]) / self.channel_stds[c]
        return normalized
    
    def crop_patch_centered_at(self, image, label, center_i, center_j):
        H, W = label.shape
        half_h = self.patch_h // 2
        half_w = self.patch_w // 2

        # Ensure the crop does not exceed image boundaries
        top = max(0, min(H - self.patch_h, center_i - half_h))
        left = max(0, min(W - self.patch_w, center_j - half_w))

        img_patch = image[top:top + self.patch_h, left:left + self.patch_w, :]
        mask_patch = label[top:top + self.patch_h, left:left + self.patch_w]
        return img_patch, mask_patch

    def get_random_patch(self, image, label):
        H, W = label.shape
        i = random.randint(0, H - self.patch_h)
        j = random.randint(0, W - self.patch_w)
        img_patch = image[i:i+self.patch_h, j:j+self.patch_w, :]
        mask_patch = label[i:i+self.patch_h, j:j+self.patch_w]
        return img_patch, mask_patch

    def __getitem__(self, idx):
        sample = self.flat_samples[idx]
        image = np.array(Image.open(sample['image_path']).convert("RGB"))  # uint8 (H, W, 3)
        label = np.array(Image.open(sample['label_path']).convert("L")).astype(np.int64)

        use_fg = random.random() < self.foreground_oversample_ratio
        fg_coords = sample['fg_coords']

        if use_fg and fg_coords:
            center_i, center_j = random.choice(fg_coords)
            img_patch, mask_patch = self.crop_patch_centered_at(image, label, center_i, center_j)
        else:
            img_patch, mask_patch = self.get_random_patch(image, label)

        if self.transform:
            transformed = self.transform(image=img_patch, mask=mask_patch)
            img_patch = transformed['image']
            mask_patch = transformed['mask']

        # Apply global Z-score normalization last
        img_patch = self.normalize_image(img_patch)

        # Normalize labels to [0, 1] for BCE loss
        if mask_patch.max() > 1.0:
            mask_patch = mask_patch / 255.0

        img_patch = torch.from_numpy(img_patch).permute(2, 0, 1)  # (C, H, W)
        mask_patch = torch.from_numpy(mask_patch).float()

        return img_patch, mask_patch

    def __len__(self):
        return len(self.flat_samples)

class OnTheFlySyntheticData(IterableDataset):
    """
    IterableDataset that yields one batch per iteration:
      (images_tensor, masks_tensor) where images_tensor.shape == (B, C, H, W)
    
    Each batch is generated by sampling a random .trk file and calling process_patches_inmemory()
    with num_patches=batch_size.
    """

    def __init__(
        self,
        trk_dir: str,
        input_nifti: str,
        white_mask_file: str,
        batch_size: int = 8,
        patch_size: Sequence[int] = (512, 1, 512),
        batches_per_epoch: int = 100,
        transform: Optional[A.Compose] = None,
        seed: Optional[int] = None,
        enable_orange_blobs: bool = True,
        orange_blob_probability: float = 0.3,
        min_streamlines_per_patch: int = 100,  # Allow sparse patches to match real data
        min_bundle_size: int = 50,  # Optimized: balance fragmentation vs sparse bundles
        voxel_size: float = 0.05,
        use_high_density_masks: bool = True,
        exclude_presets: Optional[list] = None,
    ):
        super().__init__()
        self.trk_paths = sorted(Path(trk_dir).glob("*.trk"))
        if len(self.trk_paths) == 0:
            raise ValueError(f"No .trk files found in {trk_dir}")
        self.input_nifti = input_nifti
        self.white_mask_file = white_mask_file
        self.batch_size = batch_size
        self.patch_size = list(patch_size)
        self.batches_per_epoch = batches_per_epoch
        self.transform = transform
        self.seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.enable_orange_blobs = enable_orange_blobs
        self.orange_blob_probability = orange_blob_probability
        self.min_streamlines_per_patch = min_streamlines_per_patch
        self.min_bundle_size = min_bundle_size
        self.voxel_size = voxel_size
        self.use_high_density_masks = use_high_density_masks
        self.exclude_presets = exclude_presets

    def __iter__(self):
        # Ensure different workers have different seeds
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed
        if worker_info is not None:
            worker_seed = (worker_seed + worker_info.id) % (2**32 - 1)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        for _ in range(self.batches_per_epoch):
            trk_file = str(random.choice(self.trk_paths))
            images, masks = process_patches_inmemory(
                input_nifti=self.input_nifti,
                trk_file=trk_file,
                num_patches=self.batch_size,
                patch_size=self.patch_size,
                enable_orange_blobs=self.enable_orange_blobs,
                orange_blob_probability=self.orange_blob_probability,
                min_streamlines_per_patch=self.min_streamlines_per_patch,
                min_bundle_size=self.min_bundle_size,
                voxel_size=self.voxel_size,
                white_mask_file=self.white_mask_file,
                use_high_density_masks=self.use_high_density_masks,
                exclude_presets=self.exclude_presets,
                cleanup_intermediate=True,
            )

            # images, masks assumed numpy arrays with shape (B, H, W) or (B, H, W, C)
            images = np.array(images)
            masks = np.array(masks)

            # Handle the rare case where no patches were returned
            if images.size == 0 or masks.size == 0:
                retry = 0
                max_retries = 3
                while (images.size == 0 or masks.size == 0) and retry < max_retries:
                    retry += 1
                    print(f"WARNING: process_patches_inmemory returned empty results (retry {retry}/{max_retries}) - retrying...")
                    trk_file = str(random.choice(self.trk_paths))
                    images, masks = process_patches_inmemory(
                        input_nifti=self.input_nifti,
                        trk_file=trk_file,
                        num_patches=self.batch_size,
                        patch_size=self.patch_size,
                        enable_orange_blobs=self.enable_orange_blobs,
                        orange_blob_probability=self.orange_blob_probability,
                        min_streamlines_per_patch=self.min_streamlines_per_patch,
                        min_bundle_size=self.min_bundle_size,
                        voxel_size=self.voxel_size,
                        white_mask_file=self.white_mask_file,
                        use_high_density_masks=self.use_high_density_masks,
                        exclude_presets=self.exclude_presets,
                        cleanup_intermediate=True,
                    )
                    images = np.array(images)
                    masks = np.array(masks)

                if images.size == 0 or masks.size == 0:
                    print("ERROR: process_patches_inmemory failed to produce patches after retries - skipping this batch.")
                    continue

            print('shapes 1:', images.shape, masks.shape)

            # Ensure images are channel-first (B, C, H, W)
            if images.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
                images = images[:, None, :, :]
            elif images.ndim == 4:
                # If channel-last (B, H, W, C) -> (B, C, H, W)
                if images.shape[-1] in (1, 3):
                    images = images.transpose(0, 3, 1, 2)

            images = images.astype(np.float32)
            # Normalize to [0,1] if needed
            if images.max() > 1.0:
                images = images / 255.0

            # Ensure masks are (B, 1, H, W) and binary 0/1
            if masks.ndim == 3:
                masks = masks[:, None, :, :]
            elif masks.ndim == 4 and masks.shape[-1] == 1:
                masks = masks.transpose(0, 3, 1, 2)
            masks = (masks > 0).astype(np.float32)


            # Apply albumentations per sample if provided (it works on HxW), converting back
            if self.transform is not None:
                imgs_batch, masks_batch = [], []
                for img_chw, mask_chw in zip(images, masks):
                    img_hwc = np.transpose(img_chw, (1,2,0))
                    mask_hw = mask_chw[0]

                    # If floats in [0,1], scale to 0..255 uint8 for color augmentations
                    if np.issubdtype(img_hwc.dtype, np.floating) and img_hwc.max() <= 1.0:
                        img_for_aug = (np.clip(img_hwc,0,1)*255).astype(np.uint8)
                        scale_back = True
                    else:
                        img_for_aug = img_hwc.astype(np.uint8)
                        scale_back = False
                        
                    aug = self.transform(image=img_for_aug, mask=mask_hw)
                    img_aug = np.asarray(aug['image'])
                    mask_aug = np.asarray(aug['mask'])

                    # convert back to CHW float32 (normalize to [0,1] if we scaled earlier)
                    if scale_back:
                        img_aug = img_aug.astype(np.float32) / 255.0
                    else:
                        img_aug = img_aug.astype(np.float32)

                    if img_aug.ndim == 2:
                        img_aug = img_aug[..., None]
                    img_chw_out = np.transpose(img_aug, (2,0,1)).astype(np.float32)
                    mask_chw_out = np.expand_dims((mask_aug > 0).astype(np.float32), 0)

                    imgs_batch.append(img_chw_out)
                    masks_batch.append(mask_chw_out)

                images = np.stack(imgs_batch, axis=0)
                masks = np.stack(masks_batch, axis=0)

            # convert to torch tensors
            images_t = torch.from_numpy(images).float()
            masks_t = torch.from_numpy(masks).float()

            # Normalize to 0-1 if not already (assume image dtype)
            if images_t.max() > 1.0:
                images_t = images_t / 255.0
            if masks_t.max() > 1.0:
                masks_t = masks_t / 255.0

            print('shapes 2:', images_t.shape, masks_t.shape)

            yield images_t, masks_t
    
    def __len__(self):
        # Make the dataset report the number of batches per epoch
        return int(self.batches_per_epoch)

class MixedDatasetWithOnTheFly(IterableDataset):
    """
    IterableDataset that mixes real samples with on-the-fly generated synthetic patches.
    
    Each batch contains:
    - Some samples from real data (from disk)
    - Some samples from on-the-fly synthetic generation
    """
    
    def __init__(
        self,
        real_samples: list,
        trk_dir: str,
        input_nifti: str,
        white_mask_file: str,
        synthetic_ratio: float = 0.7,
        real_transform: Optional[A.Compose] = None,
        synthetic_transform: Optional[A.Compose] = None,
        batches_per_epoch: int = 80,
        batch_size: int = 16,
        patch_size: Sequence[int] = (512, 1, 512),
        seed: int = 42,
        synthetic_output_size: Optional[int] = None,
        voxel_size: float = 0.05,
    ):
        self.real_samples = real_samples
        self.trk_dir = trk_dir
        self.input_nifti = input_nifti
        self.white_mask_file = white_mask_file
        self.synthetic_ratio = synthetic_ratio
        self.real_transform = real_transform
        self.synthetic_transform = synthetic_transform
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.seed = seed

        # Resize synthetic output to this size to match real patch size for the UNet
        self.synthetic_output_size = synthetic_output_size
        
        # Pre-compute number of synthetic/real samples per batch
        # Allow synthetic_ratio=0 to mean 0 synthetic patches (pure real data)
        self.num_synthetic_per_batch = int(batch_size * synthetic_ratio)
        self.num_real_per_batch = batch_size - self.num_synthetic_per_batch
        
        print(f"MixedDatasetWithOnTheFly: {self.num_synthetic_per_batch} synthetic (on-the-fly), {self.num_real_per_batch} real per batch")
        
        # Create random generators (seeded)
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # Create persistent real data loader with caching (like HistFinetuneDatasetCachedForeground)
        self.real_dataset = HistFinetuneDatasetCachedForeground(
            real_samples,
            patch_h=1024,  # Real patches are 1024x1024
            patch_w=1024,
            num_random_patches=1,  # We'll sample one patch at a time
            transform=real_transform,
        )
        
        # Create persistent synthetic generator only if needed (reuse across batches)
        if self.num_synthetic_per_batch > 0:
            self.synthetic_generator = OnTheFlySyntheticData(
                trk_dir=trk_dir,
                input_nifti=input_nifti,
                white_mask_file=white_mask_file,
                batch_size=self.num_synthetic_per_batch,
                patch_size=patch_size,
                batches_per_epoch=batches_per_epoch,  # Total batches needed
                transform=synthetic_transform,
                seed=seed,
                voxel_size=voxel_size,
            )
            self.synthetic_iterator = None
        else:
            print("Skipping synthetic generator initialization (synthetic_ratio=0)")
            self.synthetic_generator = None
            self.synthetic_iterator = None

    def get_synthetic_batch(self):
        """Get next synthetic batch from persistent generator, resized to match the real patch size."""
        if self.synthetic_generator is None:
            return None, None

        if self.synthetic_iterator is None:
            self.synthetic_iterator = iter(self.synthetic_generator)

        try:
            images, masks = next(self.synthetic_iterator)
        except StopIteration:
            # Restart generator when exhausted
            self.synthetic_iterator = iter(self.synthetic_generator)
            images, masks = next(self.synthetic_iterator)

        # Resize to match real patch size
        if self.synthetic_output_size is not None:
            target = self.synthetic_output_size
            if images.shape[-1] != target or images.shape[-2] != target:
                import torch.nn.functional as F
                images = F.interpolate(images, size=(target, target), mode='bilinear', align_corners=False)
                masks = F.interpolate(masks, size=(target, target), mode='nearest')

        images = self._normalize_synthetic_like_real(images)

        return images, masks

    def _normalize_synthetic_like_real(self, images: torch.Tensor) -> torch.Tensor:
        """Z-score normalize [0,1]-scale synthetic images with the same nnUNet
        channel stats used for real data, so both branches share one convention."""
        means = torch.tensor(
            self.real_dataset.channel_means, dtype=images.dtype, device=images.device
        ).view(1, -1, 1, 1)
        stds = torch.tensor(
            self.real_dataset.channel_stds, dtype=images.dtype, device=images.device
        ).view(1, -1, 1, 1)
        return (images * 255.0 - means) / stds

    def __iter__(self):
        """Generate batches of mixed real and synthetic samples."""
        # Handle multi-worker data loading: split batches across workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            batches_per_worker = int(np.ceil(self.batches_per_epoch / num_workers))
            start_batch = worker_id * batches_per_worker
            end_batch = min(start_batch + batches_per_worker, self.batches_per_epoch)
        else:
            start_batch = 0
            end_batch = int(self.batches_per_epoch)
        
        for batch_idx in range(start_batch, end_batch):
            batch_images = []
            batch_labels = []
            
            # Add synthetic samples from persistent generator (only if ratio > 0)
            if self.num_synthetic_per_batch > 0:
                try:
                    syn_images, syn_masks = self.get_synthetic_batch()
                    if syn_images is not None:
                        batch_images.append(syn_images)
                        batch_labels.append(syn_masks)
                except Exception as e:
                    print(f"Warning: Failed to generate synthetic batch: {e}")
                    # Fall back to just real data if synthetic fails
                    pass
            
            # Add real samples from cached dataset (efficient!)
            for _ in range(self.num_real_per_batch):
                # Randomly sample from real dataset (which has images cached in memory)
                idx = self.rng.randint(0, len(self.real_dataset))
                img, label = self.real_dataset[idx]
                # Real data: img is [C, H, W], label is [H, W]
                # Add batch dimension: [C, H, W] -> [1, C, H, W]
                batch_images.append(img.unsqueeze(0))
                # Add channel + batch dimensions: [H, W] -> [1, 1, H, W] to match synthetic [batch, 1, H, W]
                batch_labels.append(label.unsqueeze(0).unsqueeze(0))
            
            # Stack into batch tensors if we have samples
            if batch_images:
                batch_images = torch.cat(batch_images, dim=0)  # Concatenate along batch dimension
                batch_labels = torch.cat(batch_labels, dim=0)

                # Normalize labels to 0-1 (defensive check, should already be normalized)
                if batch_labels.max() > 1.0:
                    batch_labels = batch_labels / 255.0

                yield batch_images, batch_labels

    def __len__(self):
        """Report number of batches per epoch."""
        return int(self.batches_per_epoch)
    
