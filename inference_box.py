import os
import torch
import numpy as np
import cv2
import argparse
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io, transform
from segment_anything import sam_model_registry
from osgeo import gdal
import osam_prompt as prompt_m

# User input parameters
parser = argparse.ArgumentParser(description="Run inference on testing set based on OSAM.")
parser.add_argument("-i", "--data_path", type=str, default=r"./datasets", help="Path to the data folder.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., cuda:0, cpu).")
parser.add_argument("--fixed_box_size", type=int, default=500, help="The fixed size of the patch box prompt.")
parser.add_argument("-chk", "--checkpoint", type=str, default=r"./model/sam_vit_b_01ec64.pth", help="Path to the trained model.")
args = parser.parse_args()

# Paths setup
seg_path = os.path.join(args.data_path, "results/box")
os.makedirs(seg_path, exist_ok=True)
seg_path_binary = os.path.join(seg_path, "binary")
os.makedirs(seg_path_binary, exist_ok=True)
imgs_path = os.path.join(args.data_path, "imgs")
geojson_path = os.path.join(args.data_path, "osm/road_shps")

# Load the model
device = args.device
osam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device)
osam_model.eval()

# Get image files for testing
imgs_test_files = [f for f in os.listdir(imgs_path) if f.endswith('.tif')]

# Inference loop
for i_test in tqdm(imgs_test_files):
    i_name = os.path.basename(i_test)[:-4]
    img_np_pro = gdal.Open(os.path.join(imgs_path, i_test))
    img_np = np.transpose(img_np_pro.ReadAsArray(), (1, 2, 0))

    # Ensure the image has 3 channels
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) if len(img_np.shape) == 2 else img_np
    H, W, _ = img_3c.shape

    # Image preprocessing
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Box prompt
    json_path = os.path.join(geojson_path, f"{i_name}_geo.geojson")
    with open(json_path, 'r', encoding='UTF-8') as f:
        geojson_data = json.load(f)

    features = geojson_data.get('features', [])
    boxes_list = []
    for feature in features:
        osm_id = feature['properties']['osm_id']
        coordinates = feature['geometry']['coordinates']
        boxes_1024 = prompt_m.get_boxes_p(coordinates, img_np_pro, args.fixed_box_size, boxes_list, shift=20)

    # Transfer box coordinates to 1024x1024 scale
    box_1024 = [box_np / np.array([W, H, W, H]) * 1024 for box_np in boxes_list]
    if not box_1024:
        continue

    # Get image embedding
    with torch.no_grad():
        image_embedding = osam_model.image_encoder(img_1024_tensor)

    # Run inference with OSAM
    osam_seg, arr = osam_inference(osam_model, image_embedding, box_1024, H, W)

    # Visualize results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_np)
    for box in boxes_list:
        prompt_m.show_box(box, ax[0])
    ax[0].set_title("Input Image and Bounding Box")

    ax[1].imshow(img_np)
    ax[1].axis('off')
    if len(osam_seg.shape) == 2:
        prompt_m.show_mask_plt(osam_seg, ax[1], "red")
    else:
        for i in range(osam_seg.shape[0]):
            prompt_m.show_mask_plt(osam_seg[i], ax[1], "red")
    ax[1].set_title("Finetuned Segmentation")
    plt.savefig(os.path.join(seg_path_binary, f"ft_{i_name}_{args.fixed_box_size}px_Box.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Visualize the binary road segmentation
    background = np.zeros((1024, 1024), dtype=np.uint8)
    if len(osam_seg.shape) == 2:
        result_2 = prompt_m.show_mask_cv2(background, osam_seg)
    else:
        for i in range(osam_seg.shape[0]):
            result_2 = prompt_m.show_mask_cv2(background, osam_seg[i])
    result_2_path = os.path.join(seg_path_binary, f"ft_{i_name}_Box.png")
    cv2.imwrite(result_2_path, result_2)

print(f"Prediction by OSAM with box prompt! >> results in: {seg_path_binary}")
