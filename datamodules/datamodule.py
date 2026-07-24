import os
import sys
from pathlib import Path
from typing import Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datamodules.dataset import HistFinetuneDatasetCachedForeground, MixedDatasetWithOnTheFly
import albumentations as A

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Syntract'))


class MixedDataModule(pl.LightningDataModule):
    """
    Combines real data with on-the-fly generated synthetic data during training.
    
    synthetic_ratio: float between 0.0 and 1.0
        - 0.7 means 70% synthetic, 30% real in each training batch
    
    Training: Mixed (synthetic + real patches)
    Validation: Real data only (measure performance on target domain)
    
    Synthetic patches are generated on-the-fly, so training gets fresh samples each epoch.
    """
    def __init__(
        self,
        real_image_dir: str,
        real_label_dir: str,
        val_image_dir: str,
        val_label_dir: str,
        trk_dir: str,
        input_nifti: str,
        white_mask_file: str,
        real_transform: A.Compose,
        synthetic_transform: A.Compose,
        val_transform: A.Compose,
        synthetic_ratio: float = 0.7,
        batch_size: int = 16,
        num_workers: int = 8,
        patch_size: Sequence[int] = (512, 1, 512),
        real_patch_size: int = 1024,
        train_batches_per_epoch: int = 80,
        seed: int = 42,
        voxel_size: float = 0.05,
    ):
        super().__init__()
        
        # Real data paths (training)
        self.real_image_dir = Path(real_image_dir)
        self.real_label_dir = Path(real_label_dir)
        
        # Validation data paths (separate dataset)
        self.val_image_dir = Path(val_image_dir)
        self.val_label_dir = Path(val_label_dir)
        
        # On-the-fly synthetic generation
        self.trk_dir = trk_dir
        self.input_nifti = input_nifti
        self.white_mask_file = white_mask_file
        
        # Mixing configuration
        self.synthetic_ratio = synthetic_ratio
        self.batch_size = batch_size
        
        # Common settings
        self.num_workers = num_workers
        self.patch_size = patch_size # Size for synthetic patches
        self.real_patch_size = real_patch_size  # Size to extract from real data
        self.real_transform = real_transform  # Conservative augmentation for real data
        self.synthetic_transform = synthetic_transform  # Aggressive augmentation for synthetic data
        self.val_transform = val_transform
        self.train_batches_per_epoch = train_batches_per_epoch
        self.seed = seed
        self.voxel_size = voxel_size  # mm/px for synthetic patch generation (blockface space)

    def setup(self, stage=None):
        """
        Set up training datasets: mix real data with on-the-fly synthetic patches during training.
        Validation uses real data only to measure performance on target domain.
        """
        
        # Load real data
        real_image_files = sorted(self.real_image_dir.glob("*.png"))
        real_label_files = sorted(self.real_label_dir.glob("*.png"))
        
        if len(real_image_files) == 0:
            raise FileNotFoundError(f"No real images found in {self.real_image_dir}")
        
        real_slices = [
            {'image_path': img, 'label_path': lbl} 
            for img, lbl in zip(real_image_files, real_label_files)
        ]
        
        print(f"Loaded {len(real_slices)} real samples")
        
        # Create mixed training dataset (real + on-the-fly synthetic)
        self.train_dataset = MixedDatasetWithOnTheFly(
            real_samples=real_slices,
            trk_dir=self.trk_dir,
            input_nifti=self.input_nifti,
            white_mask_file=self.white_mask_file,
            synthetic_ratio=self.synthetic_ratio,
            real_transform=self.real_transform,
            synthetic_transform=self.synthetic_transform,
            batches_per_epoch=self.train_batches_per_epoch,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            seed=self.seed,
            synthetic_output_size=self.real_patch_size,
            voxel_size=self.voxel_size,
        )
        
        # Load separate validation data
        val_image_files = sorted(self.val_image_dir.glob("*.png"))
        val_label_files = sorted(self.val_label_dir.glob("*.png"))
        
        if len(val_image_files) == 0:
            raise FileNotFoundError(f"No validation images found in {self.val_image_dir}")
        
        val_slices = [
            {'image_path': img, 'label_path': lbl} 
            for img, lbl in zip(val_image_files, val_label_files)
        ]
        
        print(f"Loaded {len(val_slices)} validation samples from separate dataset")
        
        # Create validation dataset: REAL DATA ONLY (measure performance on target domain)
        # Extract real_patch_size x real_patch_size patches from validation data
        self.val_dataset = HistFinetuneDatasetCachedForeground(
            val_slices,
            patch_h=self.real_patch_size,
            patch_w=self.real_patch_size,
            num_random_patches=10,
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Mixed dataset returns full batches
            num_workers=self.num_workers,  # Use configurable workers (default 8)
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,  # requires num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

