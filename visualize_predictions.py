"""
This is the script for the visualization of the predicted, and ground truth fiber bundles on the
tracer data. Overlays predicted vs. ground-truth fiber-bundle boundaries on the histology slide for
a subject and saves a figure.
"""

import os
import numpy as np
from skimage import exposure
from skimage.morphology import disk, dilation
from skimage.transform import resize

from PIL import Image
import matplotlib.pyplot as plt

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

def get_boundaries(mask):
    """Extracts the boundary of a binary mask using morphological dilation."""
    mask_dilated = dilation(mask > 0, disk(5))  # Dilate with a disk of radius 5
    mask_boundary = (mask_dilated.astype(float) - (mask > 0).astype(float))
    return mask_boundary

#########################################################
# Adapt this part according to your data folder structure

subject = ''  # choose subject eid

predicted_dir = "/evaluate_results/subject/"
results_dir = predicted_dir

slides = [f"{i:03d}" for i in range(1, 36)]  # change according to your number of slides

dirname = ''
omz = zarr.open_group(dirname + '', mode='r')
dense_masks = zarr.open_group(dirname + 'masks/Fiber_dense_bundle.ome.zarr', mode='r')
moderate_masks = zarr.open_group(dirname + 'masks/Fiber_moderate_bundle.ome.zarr', mode='r')
light_masks = zarr.open_group(dirname + 'masks/Fiber_light_bundle.ome.zarr', mode='r')

level = '4'
hist_paths = np.transpose(omz[level], (1, 2, 3, 0))
label_paths_db = np.transpose(dense_masks[level], (1, 2, 3, 0))
label_paths_mb = np.transpose(moderate_masks[level], (1, 2, 3, 0))
label_paths_lb = np.transpose(light_masks[level], (1, 2, 3, 0))

num_slices = len(slides)
fig, axes = plt.subplots(nrows=num_slices, ncols=1, figsize=(20, 120))  # Adjust figsize

# Add a title for the entire figure
fig.suptitle(f'subject {subject} - results on slide', fontsize=16)
#########################################################

i = -1
for index, slide in enumerate(slides):
    i = i+1
    print(i, index, slide)

    image, boundaries = tight_crop_data(hist_paths[index])
    target_db = crop_mask_like_data(label_paths_db[index], boundaries)[:,:,0]
    target_mb = crop_mask_like_data(label_paths_mb[index], boundaries)[:,:,0]
    target_lb = crop_mask_like_data(label_paths_lb[index], boundaries)[:,:,0]
    predicted = Image.open(os.path.join(predicted_dir, f'bundle_{slide}_0000_pred.png'))
    predicted_np = np.array(predicted)

    result_mask = resize(predicted_np, (target_db.shape[0], target_db.shape[1]), preserve_range=True)
    image_resized = resize(image, (target_db.shape[0], target_db.shape[1]), preserve_range=True).astype(np.float32) / 255.0

    # Compute boundaries
    mflb_bndry = get_boundaries(target_lb)
    mfmb_bndry = get_boundaries(target_mb)
    mfdb_bndry = get_boundaries(target_db)
    result_bndry = get_boundaries(result_mask)

    # Apply adaptive histogram equalization to each channel
    img1, img2, img3 = [exposure.equalize_adapthist(image_resized[:, :, i], clip_limit=0.01) * 255.0 for i in range(3)]

    # Convert back to uint8
    img1, img2, img3 = [img.astype(np.uint8) for img in [img1, img2, img3]]

    # Fiber bundles (light - red, medium - cyan, dense - green)
    img1[mflb_bndry > 0] = 255
    img2[mflb_bndry > 0] = 0
    img3[mflb_bndry > 0] = 0

    img1[mfmb_bndry > 0] = 0
    img2[mfmb_bndry > 0] = 255
    img3[mfmb_bndry > 0] = 255

    img1[mfdb_bndry > 0] = 0
    img2[mfdb_bndry > 0] = 255
    img3[mfdb_bndry > 0] = 0

    # Results mask --> yellow
    img1[result_bndry > 0] = 255
    img2[result_bndry > 0] = 255
    img3[result_bndry > 0] = 0

    img_final = np.stack([img1, img2, img3], axis=-1)

    # Plot the slice
    axes[i].imshow(img_final)
    axes[i].set_title(f"Slice {slide}")

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(results_dir + "results_on_slide_" + subject + ".png", bbox_inches='tight', dpi=300)
