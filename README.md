# Tractography-Driven Synthetic Data Generation for Fiber Bundle Segmentation in Tracer Histology (MICCAI)

## About
This is a Pytorch implementation for the paper "Tractography-Driven Synthetic Data Generation for Fiber Bundle Segmentation in Tracer Histology" (MICCAI 2026) by Kyriaki-Margarita Bintsi, Sparsh Makharia, Yaël Balbastre, Joselyn Romero Avila, Julia F. Lehman, Suzanne N. Haber, and Anastasia Yendiki

Training mixes real and synthetic patches in each batch according to a configurable ratio
(`--synthetic_ratio`); validation always uses held-out real data only.

**This repo does not ship any data or trained checkpoints** — the commands below only run once you
point them at your own real data, registered `.trk` bundle files, and blockface NIfTI
volume (see "Data" below).

## Repository layout

- `train.py` — main training entry point (real + synthetic mixed training).
- `model.py` — `FlexibleUNet`, an nnUNet-style 2D Lightning UNet for the binary segmentation task.
- `losses.py` — `DiceBCELoss` (the loss used for all reported training runs).
- `datamodules/` — `datamodule.py` / `dataset.py`: Lightning datamodule/datasets. Real data is
  loaded from nnUNet-style paired PNG directories; synthetic data is generated on-the-fly per batch
  via Syntract (see below) with zero on-disk caching.
- `predict.py` — sliding-window inference with Gaussian-weighted patch blending and nnUNet-style
  mirroring TTA. Loads a single checkpoint by default; pass `--kfold` only if you have a 5-fold
  cross-validation ensemble to load.
- `evaluate.py` — computes the paper's reported numbers (Dice, IoU, clDice, cluster-wise
  sensitivity/FDR). Reads raw OME-Zarr histology + outline/fiber-class masks per subject. Not a
  CLI — edit the "Adapt this part" block at the top of the file for your subject/paths, then run it.
- `visualize_predictions.py` — qualitative visualization: overlays predicted vs. ground-truth
  fiber-bundle boundaries on the histology slide for a subject and saves a figure. Also not a CLI —
  same "Adapt this part" pattern.

## Setup

```bash
pip install -r requirements.txt
```

### Data synthesis (Syntract)

Synthetic patch generation is provided by [Syntract](https://github.com/Sparsh57/Syntract)
(Sparsh Makharia, LINC Team), which turns tractography (`.trk`) + a registered NIfTI volume into
dark-field-microscopy-style patches with matching fiber masks.

Clone it directly into this repo's root:

```bash
git clone https://github.com/Sparsh57/Syntract.git
pip install -r Syntract/requirements.txt
```

This gives you a `Syntract/` folder alongside `train.py`.

### Data

None of the paths below ship with this repo — supply your own:

- **Real training/validation data**: nnUNet-style directories of paired PNG images + binary masks
  (`--real_image_dir`, `--real_label_dir`, `--val_image_dir`, `--val_label_dir`).
- **Registered tractography**: a directory of `.trk` files already registered into the same space
  as your blockface volume (`--trk_dir`).
- **Blockface volume**: the registered NIfTI volume synthetic patches are extracted from
  (`--input_nifti`).
- Optionally, a white matter mask NIfTI (`--white_mask`).
- For `evaluate.py` and `visualize_predictions.py` (edited directly in each script, see below): a
  root directory containing `sub-<SUBJECT>/micr/*.ome.zarr` histology + `masks/*.ome.zarr` (Outline,
  Fiber_dense_bundle, Fiber_moderate_bundle, Fiber_light_bundle) per evaluated subject.

## Training

```bash
python train.py \
    --real_image_dir /path/to/imagesTr/ \
    --real_label_dir /path/to/labelsTr/ \
    --val_image_dir /path/to/imagesTs/ \
    --val_label_dir /path/to/labelsTs/ \
    --input_nifti /path/to/registered_blockface_volume.nii.gz \
    --synthetic_ratio 0.7 --epochs 200 --batch_size 8 --batches_per_epoch 80 \
    --checkpoint_dir results_mixed_training/my_run/ --wandb_name my_run \
    --lr 1e-4 --warmup_epochs 2
```

## Inference / evaluation

```bash
python predict.py \
    --input_folder /path/to/imagesTs_SUBJECT/ \
    --output_folder results_mixed_training/evaluate/SUBJECT/my_run/ \
    --model_folder results_mixed_training/my_run/ \
    --saved_model best
```

`evaluate.py` and `visualize_predictions.py` are not CLIs: edit the `subject`, `predicted_folder`
(or `predicted_dir`), `dirname`, and `slides` variables in the "Adapt this part" block near the top
of each file for your data, then run e.g. `python evaluate.py`.


## Citation

If you use this code, please cite our paper:
```
@article{bintsi2026tractography,
  title={Tractography-Driven Synthetic Data Generation for Fiber Bundle Segmentation in Tracer Histology},
  author={Bintsi, Kyriaki-Margarita and Makharia, Sparsh and Balbastre, Ya{\"e}l and Avila, Joselyn Romero and Lehman, Julia F and Haber, Suzanne N and Yendiki, Anastasia},
  journal={arXiv preprint arXiv:2606.26898},
  year={2026}
}
```
and
[Syntract](https://github.com/Sparsh57/Syntract) for the data synthesis pipeline.

