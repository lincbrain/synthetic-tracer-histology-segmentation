"""
Calculation of the metrics sensitivity (TPR), false positives, FDR, Dice score, IoU, and clDice
for a subject's predicted fiber bundles against the ground-truth OME-Zarr masks.
"""

import os
import numpy as np
from PIL import Image
from skimage.measure import label
from skimage.morphology import skeletonize
import pandas as pd
import zarr


def cut_zeros1d(im_array):
    """
    Find the window for cropping the data closer to the brain
    :param im_array: input array
    :return: starting and end indices, and length of non-zero intensity values
    """
    im_list = list(im_array > 0)
    start_index = im_list.index(1)
    end_index = im_list[::-1].index(1)
    length = len(im_array[start_index:])-end_index
    return start_index, end_index, length

def tight_crop_data(img_data):
    """
    Crop the data tighter to the brain
    :param img_data: input array
    :return: cropped image and the bounding box coordinates and dimensions.
    """
    row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
    col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
    stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)
    rsid, reid, rlen = cut_zeros1d(row_sum)
    csid, ceid, clen = cut_zeros1d(col_sum)
    ssid, seid, slen = cut_zeros1d(stack_sum)
    return img_data[rsid:rsid+rlen, csid:csid+clen, ssid:ssid+slen], [rsid, rlen, csid, clen, ssid, slen]

def crop_mask_like_data(mask, bounding_box):
    """
    Crop the masks to same dimensions with the data
    :param mask: mask that needs to be cropped
    :param bounding box: the bounding box coordinates and dimensions.
    :return: cropped mask
    """
    rsid, rlen, csid, clen, ssid, slen = bounding_box
    cropped_mask = mask[rsid:rsid+rlen, csid:csid+clen]
    return cropped_mask

def calculate_iou(pred, target):
    """
    Calculate the Intersection over Union (IoU) between predicted and target masks.

    Parameters:
    - pred: Predicted binary segmentation map (numpy array)
    - target: Ground truth binary segmentation map (numpy array)

    Returns:
    - iou: The IoU score between pred and target
    """
    # Ensure the input maps are numpy arrays and binary
    pred = np.asarray(pred > 0, dtype=np.float32)
    target = np.asarray(target > 0, dtype=np.float32)

    # Calculate intersection and union
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection

    # Calculate IoU
    if union == 0:
        # If both pred and target are empty, return 1.0
        # If only one is empty, return 0.0
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou

def calculate_class_wise_iou(pred, target):
    """
    Calculate class-wise IoU for dense, moderate, and light fiber bundles.

    Parameters:
    - pred: Predicted segmentation map (numpy array)
    - target: Ground truth segmentation map with classes [1,2,3] (numpy array)

    Returns:
    - iou_dense: IoU for dense fibers (class 3)
    - iou_moderate: IoU for moderate fibers (class 2)
    - iou_light: IoU for light fibers (class 1)
    """
    ious = []

    # Calculate IoU for each class
    for class_id in [1, 2, 3]:  # light, moderate, dense
        # Create binary masks for the current class
        pred_class = (pred == class_id).astype(np.float32)
        target_class = (target == class_id).astype(np.float32)

        # Calculate IoU for this class
        intersection = np.sum(pred_class * target_class)
        union = np.sum(pred_class) + np.sum(target_class) - intersection

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    return tuple(ious)  # returns (iou_light, iou_moderate, iou_dense)

def get_clusterwise_metrics_typewise(pred, col_target):
    sens = []
    for ct in [1, 2, 3]:
        labeltarget, num_target = label(col_target == ct, return_num=True)
        tp = np.setdiff1d(np.union1d(labeltarget[pred > 0], []), 0)
        tp = len(list(tp))
        fn = num_target - tp
        if (tp + fn) == 0:
            sens_val = 1
        else:
            sens_val = tp / (tp + fn)
        sens.append(sens_val)
    sens_light, sens_mod, sens_dense = sens

    labelpred, num_pred = label(pred > 0, return_num=True)
    tp = np.setdiff1d(np.union1d(labelpred[col_target > 0], []), 0)
    tp = len(list(tp))
    fp = num_pred - tp
    if num_pred == 0:
        fdr = 0
    else:
        fdr = fp / num_pred

    return sens_dense, sens_mod, sens_light, tp, fp, fdr

def dice_score(pred, target):
    """
    Calculate Dice score for binary classification.
    pred: binary prediction [0,1]
    target: will be converted to binary [0,1] where classes [1,2,3] become 1
    """
    # Ensure binary prediction
    pred_binary = (pred > 0).astype(np.float32)

    # Convert multi-class target to binary
    target_binary = (target > 0).astype(np.float32)  # This automatically maps [1,2,3] to 1

    # Calculate intersection and union
    intersection = np.sum(pred_binary * target_binary)
    sum_pred = np.sum(pred_binary)
    sum_target = np.sum(target_binary)

    # Calculate Dice
    dice = 2 * intersection / (sum_pred + sum_target + 1e-6)

    return dice

def compute_cldice(pred, target, smooth=1e-6):
    """
    Compute the centerline Dice (clDice) similarity between two 2D binary masks.

    Returns a value in [0,1] where 1.0 is perfect agreement.
    """
    # Convert to binary
    pred_bin = (pred > 0).astype(np.uint8)
    target_bin = (target > 0).astype(np.uint8)

    # Both empty -> perfect
    if pred_bin.sum() == 0 and target_bin.sum() == 0:
        return 1.0

    # Skeletonize (works on boolean 2D arrays)
    try:
        skel_pred = skeletonize(pred_bin > 0)
        skel_true = skeletonize(target_bin > 0)
    except Exception:
        # If skeletonize fails for some reason, fall back to simple Dice on foreground
        inter = np.sum(pred_bin * target_bin)
        denom = pred_bin.sum() + target_bin.sum()
        return 2.0 * inter / denom if denom > 0 else 1.0

    # True precision and true sensitivity
    tprec = (np.sum(skel_pred & (target_bin > 0)) + smooth) / (np.sum(skel_pred) + smooth)
    tsens = (np.sum(skel_true & (pred_bin > 0)) + smooth) / (np.sum(skel_true) + smooth)

    # Combine into clDice similarity (higher is better)
    denom = (tprec + tsens)
    if denom == 0:
        return 0.0
    cldice = 2.0 * (tprec * tsens) / denom
    return float(cldice)

#########################################################
# Adapt this part according to your data folder structure

subject = ''  # choose subject eid
half_brain = False  # set True to calculate metrics only on the left half of the image

dirname = ''
omz = zarr.open_group(dirname + '', mode='r')
outline = zarr.open_group(dirname + '/masks/Outline.ome.zarr', mode='r')
dense_masks = zarr.open_group(dirname + '/masks/Fiber_dense_bundle.ome.zarr', mode='r')
moderate_masks = zarr.open_group(dirname + '/masks/Fiber_moderate_bundle.ome.zarr', mode='r')
light_masks = zarr.open_group(dirname + '/masks/Fiber_light_bundle.ome.zarr', mode='r')

level = '4'
hist_paths = np.transpose(omz[level], (1, 2, 3, 0))
brain_paths = np.transpose(outline[level], (1, 2, 3, 0))
label_paths_db = np.transpose(dense_masks[level], (1, 2, 3, 0))
label_paths_mb = np.transpose(moderate_masks[level], (1, 2, 3, 0))
label_paths_lb = np.transpose(light_masks[level], (1, 2, 3, 0))

predicted_folder = "/evaluate_results/unet_ensemble/subject/"
results_dir = predicted_folder

slides = [f"{i:03d}" for i in range(1, 36)]  # change according to your number of slides
#########################################################
results = []

for ix, slide in enumerate(slides):
    print(ix, slide)
    index = ix

    image, boundaries = tight_crop_data(hist_paths[index])
    mask = crop_mask_like_data(brain_paths[index], boundaries)[:,:,0]

    # Get masks
    target_db = crop_mask_like_data(label_paths_db[index], boundaries)[:,:,0]
    target_mb = crop_mask_like_data(label_paths_mb[index], boundaries)[:,:,0]
    target_lb = crop_mask_like_data(label_paths_lb[index], boundaries)[:,:,0]

    predicted = Image.open(os.path.join(predicted_folder, f'bundle_{slide}_0000_pred.png'))
    predicted_np = np.array(predicted)

    # Normalize to binary (0,1)
    predicted_np = (predicted_np > 127).astype(np.uint8)

    target_map = np.zeros_like(target_db)
    target_map[target_lb > 0] = 1
    target_map[target_mb > 0] = 2
    target_map[target_db > 0] = 3
    target_map = target_map.astype(int)

    # If half_brain is set, crop to left half
    if half_brain:
        half_w = target_map.shape[1] // 2
        target_map = target_map[:, :half_w]
        mask = mask[:, :half_w]
        predicted_np = predicted_np[:, :half_w]

    # Apply the mask to consider only the region of interest (ROI)
    target = target_map * mask
    pred = predicted_np * mask

    # Compute cluster-wise metrics
    sensds, sensms, sensls, tps, fps, fdrs = get_clusterwise_metrics_typewise(pred, target)

    score = dice_score(pred, target)

    iou_score = calculate_iou(pred, target)
    iou_light, iou_moderate, iou_dense = calculate_class_wise_iou(pred, target)
    cldice_score = compute_cldice(pred, target)

    result = {
            'file': slide,
            'sensitivity_dense': sensds,
            'sensitivity_moderate': sensms,
            'sensitivity_light': sensls,
            'false_positives': fps,
            'true_positives': tps,
            'fdr': fdrs,
            'dice_score': score,
            'iou_score': iou_score,
            'iou_dense': iou_dense,
            'iou_moderate': iou_moderate,
            'iou_light': iou_light,
            'cldice': cldice_score,
        }
    results.append(result)

# Save the results to a CSV file
df = pd.DataFrame(results)
csv_name = "bundle_evaluation_results_inside_mask_half_brain.csv" if half_brain else "bundle_evaluation_results_inside_mask.csv"
df.to_csv(os.path.join(results_dir, csv_name), index=False)
