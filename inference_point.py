import os
import torch
import numpy as np
import cv2
import argparse
import datetime
import matplotlib.pyplot as plt
from skimage import io, transform
from segment_anything import sam_model_registry
import osam_prompt as prompt_m
from tqdm import tqdm

# User input parameters
parser = argparse.ArgumentParser(description="Run inference on the testing set based on OSAM.")
parser.add_argument("-i", "--data_path", type=str, default=r"./datasets", help="Path to the data folder.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cuda:0, cpu).")
parser.add_argument("--sample_mode", type=str, default="grid", help="Sample mode of points for prompt input.")
parser.add_argument("-chk", "--checkpoint", type=str, default=r"./model/sam_vit_b_01ec64.pth", help="Path to the trained model.")
args = parser.parse_args()

# Paths setup
seg_path = os.path.join(args.data_path, "results/points")
os.makedirs(seg_path, exist_ok=True)
seg_path_binary = os.path.join(seg_path, "binary")
os.makedirs(seg_path_binary, exist_ok=True)
imgs_path = os.path.join(args.data_path, "imgs")
labs_path = os.path.join(args.data_path, "osm/road_line")

# Load the model
device = args.device
osam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device)
osam_model.eval()

# Get image files for testing
imgs_test_files = [f for f in os.listdir(imgs_path) if f.endswith('.tif')]

# Inference loop
for i_test in tqdm(imgs_test_files):
    start_time = datetime.datetime.now()
    i_name = os.path.basename(i_test)[:-4]
    img_np = io.imread(os.path.join(imgs_path, i_test))

    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np

    H, W, _ = img_3c.shape

    # Image preprocessing
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Point prompt
    lab_path = os.path.join(labs_path, f"{i_name}_line.tif")
    lab_np = io.imread(lab_path)
    if lab_np.max() == 0:
        continue

    coords_np = np.array(prompt_m.find_pixels(lab_path, args.sample_mode))
    points_num = len(coords_np)
    labels_np = np.ones(points_num)

    # Transfer coordinates to 1024x1024 scale
    scale_x, scale_y = 1024 / H, 1024 / W
    coords_1024 = coords_np * np.array([scale_x, scale_y])

    # Get image embedding
    with torch.no_grad():
        image_embedding = osam_model.image_encoder(img_1024_tensor)

    # Run inference with OSAM
    osam_seg, arr = osam_inference(osam_model, image_embedding, coords_1024, labels_np, H, W)
    end_time = datetime.datetime.now()
    running_time = (end_time - start_time).seconds

    # Visualization: Red road and points on original image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_np)
    prompt_m.show_points(coords_np, labels_np, ax[0])
    ax[0].set_title(f"Input Image and {points_num} Points prompt | time {running_time} s")
    ax[1].imshow(img_np)
    ax[1].axis('off')
    if len(osam_seg.shape) == 2:
        prompt_m.show_mask_plt(osam_seg, ax[1], "red")
    else:
        for i in range(osam_seg.shape[0]):
            prompt_m.show_mask_plt(osam_seg[i], ax[1], "red")
    ax[1].set_title("Finetuned Segmentation")
    plt.savefig(os.path.join(seg_path, f"ft_{i_name}_{points_num}Points.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Visualize binary road segmentation
    background = np.zeros((1024, 1024), dtype=np.uint8)
    if len(osam_seg.shape) == 2:
        result_2 = prompt_m.show_mask_cv2(background, osam_seg)
    else:
        for i in range(osam_seg.shape[0]):
            result_2 = prompt_m.show_mask_cv2(background, osam_seg[i])
    result_2_path = os.path.join(seg_path_binary, f"ft_{i_name}_Points{points_num}.png")
    cv2.imwrite(result_2_path, result_2)

print(f"Prediction by OSAM with point prompt! >> results in: {seg_path_binary}")
