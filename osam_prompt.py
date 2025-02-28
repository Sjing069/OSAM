import random
import numpy as np
import cv2
from PIL import Image
from osgeo import gdal
import matplotlib.pyplot as plt

def find_pixels(lab_path, sample_mode):
    """
    Find pixel coordinates in a label image (where white pixels represent roads).
    Supports random, interval, or grid sampling.
    """
    image = Image.open(lab_path).split()[0]
    width, height = image.size
    all_pixels = [(x, y) for x in range(width) for y in range(height) if image.getpixel((x, y)) == 255]

    if sample_mode == "random":
        return random.sample(all_pixels, 20)
    elif sample_mode == "interval":
        return all_pixels[::50]
    elif sample_mode == "grid":
        return grid_sampling(all_pixels, grid_size=(100, 100))
    else:
        raise ValueError("Invalid sampling mode. Choose 'random', 'interval', or 'grid'.")

def grid_sampling(points, grid_size=(10, 10)):
    """
    Perform grid sampling on a set of points. For each grid cell, select the first point found.
    """
    max_x, max_y = np.max(points, axis=0)
    sampled_points = []
    for x in range(0, max_x, grid_size[0]):
        for y in range(0, max_y, grid_size[1]):
            cell_points = [p for p in points if x <= p[0] < x + grid_size[0] and y <= p[1] < y + grid_size[1]]
            if cell_points:
                sampled_points.append(cell_points[0])
    return sampled_points

def show_points(coords, labels, ax, marker_size=10):
    """
    Plot positive and negative points on a matplotlib axis.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='white', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """
    Draw a bounding box on a matplotlib axis.
    """
    x, y, x2, y2 = box
    width, height = x2 - x, y2 - y
    ax.add_patch(plt.Rectangle((x, y), width, height, edgecolor='yellow', facecolor='none', lw=2))

def show_mask_plt(mask, ax, custom_color=None):
    """
    Display a mask with a specified color on a matplotlib axis.
    If no color is specified, a random color is used.
    """
    if custom_color is None:
        color = np.concatenate([np.random.random(3), [0.6]])
    elif custom_color == "red":
        color = np.array([1, 30/255, 30/255, 0.7])
    elif custom_color == "white":
        color = np.array([1, 1, 1, 1])
    elif custom_color == "yellow":
        color = np.array([1, 1, 0, 1])
    else:
        color = np.concatenate([np.random.random(3), [0.6]])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)

def show_mask_cv2(base_img, mask):
    """
    Overlay a binary mask on an image using OpenCV.
    """
    base_img[mask > 0] = 255
    return base_img

def patch_boxes(start_point, end_point, fixed_box_size, shift, img_width, img_height):
    """
    Create a series of patches (bounding boxes) along a road segment.
    """
    x1, y1 = start_point
    x2, y2 = end_point
    road_width = abs(x2 - x1)
    road_height = abs(y2 - y1)
    boxes = []

    if road_width < fixed_box_size and road_height < fixed_box_size:
        boxes.append([x1, y1, x2, y2])
    else:
        if road_width >= road_height:
            num_boxes = road_width // fixed_box_size
            x_box_size = fixed_box_size
            y_box_size = road_height / num_boxes if num_boxes else road_height
            for i in range(int(num_boxes)):
                bx1 = x1 + i * x_box_size
                by1 = int(max(0, y1 + i * y_box_size - shift))
                bx2 = x1 + (i + 1) * x_box_size
                by2 = int(min(img_width, y1 + (i + 1) * y_box_size + shift))
                boxes.append([bx1, by1, bx2, by2])
            # Handle remainder if exists
            if road_width % fixed_box_size > 0:
                bx1 = x1 + int(num_boxes) * x_box_size
                boxes.append([bx1, int(y1 + num_boxes * y_box_size - shift), x2, int(y2 + shift)])
        else:
            num_boxes = road_height // fixed_box_size
            y_box_size = fixed_box_size
            x_box_size = road_width / num_boxes if num_boxes else road_width
            for i in range(int(num_boxes)):
                bx1 = int(max(0, x1 + i * x_box_size - shift))
                by1 = y1 + i * y_box_size
                bx2 = int(min(img_height, x1 + (i + 1) * x_box_size + shift))
                by2 = y1 + (i + 1) * y_box_size
                boxes.append([bx1, by1, bx2, by2])
            if road_height % fixed_box_size > 0:
                bx1 = int(max(0, x1 + num_boxes * x_box_size - shift))
                boxes.append([bx1, y1 + int(num_boxes * y_box_size), x2 + shift, y2])
    return boxes


def get_boxes_p(lonlat_geojson, image, fixed_box_size, shift):
    """
    Generate patch boxes for a road using its geographic coordinates.
    Chooses the leftmost point as the starting point.
    """
    img_width, img_height = image.RasterXSize, image.RasterYSize
    lonlat_geojson = np.asarray(lonlat_geojson, dtype=object)

    if lonlat_geojson.ndim == 2:
        a_point = lonlat2pixel(image, lonlat_geojson[0])
        b_point = lonlat2pixel(image, lonlat_geojson[-1])
        start_point, end_point = (a_point, b_point) if a_point[0] <= b_point[0] else (b_point, a_point)
        return patch_boxes(start_point, end_point, fixed_box_size, shift, img_width, img_height)
    else:
        # For multiple road segments, use the first and last segment endpoints
        a_point = lonlat2pixel(image, lonlat_geojson[0][0])
        b_point = lonlat2pixel(image, lonlat_geojson[-1][-1])
        start_point, end_point = (a_point, b_point) if a_point[0] <= b_point[0] else (b_point, a_point)
        return patch_boxes(start_point, end_point, fixed_box_size, shift, img_width, img_height)

def lonlat2pixel(image, lonlat):
    """
    Convert geographic coordinates (longitude, latitude) to pixel coordinates.
    """
    trans = image.GetGeoTransform()
    matrix = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    offset = np.array([lonlat[0] - trans[0], lonlat[1] - trans[3]])
    pixel_coords = np.linalg.solve(matrix, offset)
    return [int(pixel_coords[0]), int(pixel_coords[1])]

def reference_to_sam_mask(ref_mask: np.ndarray, threshold: int = 127, pad_all_sides: bool = False) -> np.ndarray:
    """
    Convert a grayscale mask to a binary mask, resize it to have its longest side equal to 256,
    pad it to be square, and add a channel dimension.
    """
    binary_mask = (ref_mask > threshold).astype(np.uint8)
    resized_mask, new_height, new_width = resize_mask(binary_mask)
    square_mask = pad_mask(resized_mask, new_height, new_width, pad_all_sides)
    return np.expand_dims(square_mask, axis=0)

def resize_mask(ref_mask: np.ndarray, longest_side: int = 256):
    """
    Resize an image so that its longest side equals the specified value.
    Returns the resized image, new height, and new width.
    """
    height, width = ref_mask.shape[:2]
    if height > width:
        new_height = longest_side
        new_width = int(width * longest_side / height)
    else:
        new_width = longest_side
        new_height = int(height * longest_side / width)
    resized = cv2.resize(ref_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return resized, new_height, new_width

def pad_mask(ref_mask: np.ndarray, new_height: int, new_width: int, pad_all_sides: bool = False) -> np.ndarray:
    """
    Pad an image to a fixed size (256x256) to make it square.
    If pad_all_sides is True, pads equally on all sides; otherwise pads bottom and right.
    """
    pad_height = 256 - new_height
    pad_width = 256 - new_width
    if pad_all_sides:
        padding = (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
        )
    else:
        padding = ((0, pad_height), (0, pad_width))
    return np.pad(ref_mask, padding, mode="constant")


def enlarge_box(original_box, enlarge_px, image):
    """
    Enlarge a bounding box by a specified number of pixels, ensuring it stays within the image bounds.
    original_box: [x1, y1, x2, y2]
    """
    width, height = image.RasterXSize, image.RasterYSize
    x1, y1, x2, y2 = original_box
    x1 = max(0, x1 - enlarge_px)
    y1 = max(0, y1 - enlarge_px)
    x2 = min(width, x2 + enlarge_px)
    y2 = min(height, y2 + enlarge_px)
    return [x1, y1, x2, y2]


def find_box(lonlat_geojson, image, enlarge_px, is_subsection):
    """
    Generate bounding boxes for road segments based on geographic coordinates.
    If is_subsection is True, generates a box for each road segment.
    Otherwise, creates one box for the entire road.
    """
    boxes = []
    lonlat_geojson = np.asarray(lonlat_geojson, dtype=object)

    if lonlat_geojson.ndim == 2:
        # Single road: use first and last coordinates.
        start = lonlat2pixel(image, lonlat_geojson[0])
        end = lonlat2pixel(image, lonlat_geojson[-1])
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])
        box = enlarge_box([x1, y1, x2, y2], enlarge_px, image)
        boxes.append(box)
    else:
        if is_subsection:
            for segment in lonlat_geojson:
                start = lonlat2pixel(image, segment[0])
                end = lonlat2pixel(image, segment[-1])
                x1, y1 = min(start[0], end[0]), min(start[1], end[1])
                x2, y2 = max(start[0], end[0]), max(start[1], end[1])
                box = enlarge_box([x1, y1, x2, y2], enlarge_px, image)
                boxes.append(box)
        else:
            start = lonlat2pixel(image, lonlat_geojson[0][0])
            end = lonlat2pixel(image, lonlat_geojson[-1][-1])
            x1, y1 = min(start[0], end[0]), min(start[1], end[1])
            x2, y2 = max(start[0], end[0]), max(start[1], end[1])
            box = enlarge_box([x1, y1, x2, y2], enlarge_px, image)
            boxes.append(box)
    return boxes
