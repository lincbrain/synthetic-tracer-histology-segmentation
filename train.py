import argparse
import numpy as np
import os
import wandb
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Syntract'))

torch.set_float32_matmul_precision('high')
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl

from datamodules.datamodule import MixedDataModule
from model import *
import albumentations as A

print(torch.cuda.is_available())

"""
Mixed training: combine real and synthetic patches during training.

Synthetic patches are generated on-the-fly, so both train and val get fresh samples each epoch.
Real and synthetic patches are mixed according to synthetic_ratio (default 70% synthetic, 30% real).

Example usage:
python train.py \
    --real_image_dir /path/to/imagesTr/ \
    --real_label_dir /path/to/labelsTr/ \
    --val_image_dir /path/to/imagesTs/ \
    --val_label_dir /path/to/labelsTs/ \
    --input_nifti /path/to/registered_blockface_volume.nii.gz \
    --synthetic_ratio 0.7 --epochs 200 --batch_size 8 --batches_per_epoch 80 \
    --checkpoint_dir results_mixed_training/my_run/ --wandb_name my_run \
    --lr 1e-4 --warmup_epochs 2
"""

def get_args_parser():
    parser = argparse.ArgumentParser('Mixed training (Real + Synthetic)', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batches_per_epoch', default=80, type=int)
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help="Directory to save checkpoints")
    parser.add_argument('--wandb_name', type=str, required=True,
                    help='Name to save on the wandb run')

    parser.add_argument('--synthetic_ratio', type=float, default=0.7,
                        help='Ratio of synthetic patches in training batch (0.0-1.0). E.g., 0.7 = 70% synthetic, 30% real')
    parser.add_argument('--shot', type=str, default=None,
                        help='Optional free-text label for logging/naming only (e.g. "1", "5", "whole"). Does not affect paths.')

    # Real data (nnUNet-style paired image/label PNG directories)
    parser.add_argument('--real_image_dir', type=str, required=True, help='Directory of real training images')
    parser.add_argument('--real_label_dir', type=str, required=True, help='Directory of real training labels')
    parser.add_argument('--val_image_dir', type=str, required=True, help='Directory of real validation images')
    parser.add_argument('--val_label_dir', type=str, required=True, help='Directory of real validation labels')

    # Synthetic data generation (Syntract)
    parser.add_argument('--trk_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'registered_trk_files'),
                        help='Directory of registered .trk tractography bundle files')
    parser.add_argument('--input_nifti', type=str, required=True,
                        help='Path to the registered blockface NIfTI volume used for synthetic patch generation')

    parser.add_argument('--pos_weight', type=float, default=1.0,
                        help='Weight for positive class (bundle). <1.0 = penalize over-prediction, >1.0 = penalize under-prediction')
    parser.add_argument('--white_mask', type=str, default=None,
                        help='Path to white matter mask file (omit to disable)')

    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Number of warmup epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder layers during training')

    # other parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')   
    parser.set_defaults(pin_mem=True)

    return parser 

class LogTrainSamplesCallback(pl.Callback):
    """
    Logs the first training batch of epoch 0 to wandb (once only).
    First `num_synthetic` images per batch are synthetic, the rest real.
    """
    def __init__(self, synthetic_ratio: float):
        self.synthetic_ratio = synthetic_ratio
        # nnUNet Z-score stats used to normalize both real and synthetic patches
        self._means = np.array([26.370181, 28.796626, 25.990918], dtype=np.float32)
        self._stds  = np.array([31.776849, 28.548427, 16.703267], dtype=np.float32)

    def _denorm(self, img_chw):
        """(C,H,W) float tensor → uint8 numpy (H,W,3)"""
        arr = img_chw.cpu().float().permute(1, 2, 0).numpy()  # (H,W,3)
        arr = arr * self._stds + self._means
        return np.clip(arr, 0, 255).astype(np.uint8)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.current_epoch != 0 or batch_idx != 0:
            return
        if trainer.logger is None:
            return

        x, y = batch   # x: (B,3,H,W) normalised,  y: (B,1,H,W) or (B,H,W)
        B = x.shape[0]
        num_synthetic = int(B * self.synthetic_ratio)

        wandb_images = []
        for i in range(B):
            rgb = self._denorm(x[i])            # (H,W,3) uint8
            msk = y[i].squeeze().cpu().float().numpy()  # (H,W)

            # Overlay mask in red
            overlay = rgb.copy()
            fg = msk > 0.5
            overlay[fg, 0] = 220
            overlay[fg, 1] = (overlay[fg, 1] // 2).astype(np.uint8)
            overlay[fg, 2] = (overlay[fg, 2] // 2).astype(np.uint8)

            label = f"{'SYNTH' if i < num_synthetic else 'REAL'} [{i}] fg={msk.mean():.3f}"
            wandb_images.append(wandb.Image(overlay, caption=label))

        trainer.logger.experiment.log(
            {f"train_samples/epoch_{trainer.current_epoch:03d}": wandb_images},
            step=trainer.global_step,
        )


def main(args):
    print(f"Starting mixed training: {args.synthetic_ratio*100:.0f}% synthetic, {(1-args.synthetic_ratio)*100:.0f}% real")
    print('Experiment saved here', args.checkpoint_dir)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Fixed random seeds
    seed_everything(args.seed, workers=True)

    # Real data: Conservative geometric, but stronger appearance augmentation for appearance invariance
    real_transform = A.Compose([
        A.Resize(height=1024, width=1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.5),
    ])

    # Synthetic data: Stronger geometric and appearance augmentation, plus moderate staining simulation and noise
    synthetic_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ElasticTransform(p=0.3, alpha=50, sigma=5),
        
        # Moderate appearance augmentation - strong enough to help, but won't cause NaN
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        
        # Moderate staining simulation
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        
        # Moderate noise
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.6),
        
        # Light blur
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
    ])
    
    # Validation transform: EMPTY (real data already 1024x1024, no augmentation)
    val_transform = A.Compose([])

    if args.shot is not None:
        print(f"Using real-data regime labeled '{args.shot}'")

    white_mask = None if (args.white_mask is None or args.white_mask.lower() == "none") else args.white_mask
    patch_size = [512, 1, 512]  # Generate synthetic patches at 512x512, resized to real_patch_size before mixing
    # Match physical field of view to the real data: real patches are 1024px cropped at
    # 7.035424 um/px --> 7.204 mm FOV. With patch_size=512,
    # voxel_size = 7.204 / 512 mm/px gives synthetic patches the same physical FOV.
    voxel_size = (1024 * 0.007035424) / 512

    datamodule = MixedDataModule(
        real_image_dir=args.real_image_dir,
        real_label_dir=args.real_label_dir,
        val_image_dir=args.val_image_dir,
        val_label_dir=args.val_label_dir,
        trk_dir=args.trk_dir,
        input_nifti=args.input_nifti,
        white_mask_file=white_mask,
        synthetic_ratio=args.synthetic_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=patch_size,
        real_patch_size=1024,
        voxel_size=voxel_size,
        seed=args.seed,
        real_transform=real_transform,
        synthetic_transform=synthetic_transform,
        val_transform=val_transform,
        train_batches_per_epoch=args.batches_per_epoch,
    )

    # checkpoint name
    checkpoint_name = "last.ckpt"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, starting from scratch")

    # wandb logging
    wandb_logger = WandbLogger(
        project="unet-training-mixed",
        name=args.wandb_name,
        save_code=False,
        log_model=False
    )

    model = FlexibleUNet(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        freeze_encoder=args.freeze_encoder)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best_mixed_unet-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,           
        save_top_k=1       
    )

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Log sample training images to wandb for the first batch of epoch 0 only
    log_samples_callback = LogTrainSamplesCallback(
        synthetic_ratio=args.synthetic_ratio,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        precision=16,
        callbacks=[checkpoint_callback, lr_monitor, log_samples_callback],
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Train
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        print("Starting training from scratch")
        trainer.fit(model, datamodule=datamodule)
    
    # Finish the current WandB run
    wandb.finish()
        

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()  
    main(args) 
