import os
import re
from typing import List
from PIL import Image
import numpy as np
import pickle
from ultralytics import YOLO


def ensure_directory_exists(directory_path):
    """Creates the directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def get_file_names(directory_path):

    # Get all file names in the directory
    file_names = os.listdir(directory_path)

    # Optionally, store the file names in a list
    file_list = [file_name for file_name in file_names]
    print(f"There are {len(file_names)} images to analyse")
    return file_list


def combine_channels(channel1, channel2):
    """
    Combine two grayscale channels into an RGB image by keeping
    the third channel as zero (black).
    """
    combined = np.zeros((channel1.shape[0], channel1.shape[1], 3), dtype=np.uint8)
    combined[:, :, 0] = channel1  # Assign to Red
    combined[:, :, 1] = channel2  # Assign to Green
    return combined


def save_color_channel(channel, color, filename):
    """
    Saves a single-channel image as a full RGB image in the desired color.
    """
    directory = os.path.dirname(filename)
    ensure_directory_exists(directory)

    h, w = channel.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    if color == "R":
        color_image[:, :, 0] = channel
    elif color == "G":
        color_image[:, :, 1] = channel
    elif color == "B":
        color_image[:, :, 2] = channel

    Image.fromarray(color_image).save(filename, "JPEG")


def rgb_images(image, filename_prefix, experiment_name):
    """
    Saves R, G, B channels and combinations.
    """
    rgb_dir = os.path.join(experiment_name, "rgb")
    ensure_directory_exists(rgb_dir)

    rgb_array = np.array(image)

    R = rgb_array[:, :, 0]
    G = rgb_array[:, :, 1]
    B = rgb_array[:, :, 2]

    # Save each channel as a color image
    save_color_channel(R, "R", f"{experiment_name}/rgb/{filename_prefix}_R.jpg")
    save_color_channel(G, "G", f"{experiment_name}/rgb/{filename_prefix}_G.jpg")
    save_color_channel(B, "B", f"{experiment_name}/rgb/{filename_prefix}_B.jpg")

    # Combine channels
    RG = combine_channels(R, G)  # Red + Green, no Blue
    GB = combine_channels(G, B)  # Green + Blue, no Red
    RB = combine_channels(R, B)  # Red + Blue, no Green

    # Save the combined images
    Image.fromarray(RG).save(f"{experiment_name}/rgb/{filename_prefix}_RG.jpg", "JPEG")
    Image.fromarray(GB).save(f"{experiment_name}/rgb/{filename_prefix}_GB.jpg", "JPEG")
    Image.fromarray(RB).save(f"{experiment_name}/rgb/{filename_prefix}_RB.jpg", "JPEG")
    return R, G, B, RG, GB, RB, rgb_array


def compress_channel(channel, k):
    """Applies SVD to compress a single color channel."""
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    total_variance = np.sum(S)  # Sum of all singular values
    retained_variance = np.sum(S[:k])  # Sum of top-k singular values
    explained_variance_ratio = retained_variance / total_variance
    # print(f"Valores singulares:{S.shape}") --> son 1920 en este caso!
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])
    compressed = U[:, :k] @ S_k @ Vt[:k, :]
    return np.clip(compressed, 0, 255), explained_variance_ratio


# Function to compress the full image
def compress_image_rgb(img, k):
    variance_ratio = {}
    """Compress an RGB image by applying SVD to each channel separately."""
    compressed_channels = []
    for i in range(3):
        c, var = compress_channel(img[:, :, i], k)
        compressed_channels.append(c)
        variance_ratio[i] = {"k": k, "variance_ratio": var}
    return np.stack(compressed_channels, axis=2).astype(np.uint8), variance_ratio


def preprocessing_images(
    base_path: str,
    image_names: List[str],
    experiment_name="ooid_general_experiment",
    N=50,
):
    """
    Image preprocessing.
    base_path (str): images base path
    image_names (List[str]): image names
    experiment_name (str, optional): Experiment name. Defaults to "ooid_general_experiment".
    """

    svd_dir = os.path.join(experiment_name, "svd")
    ensure_directory_exists(svd_dir)

    k_values = list(range(1, 150, N)) + [1000]
    variance_ratios = []
    for i in range(len(image_names)):
        path = base_path + image_names[i]
        print(f"Imagen {path} siendo procesada {i}/{len(image_names)}")
        image = Image.open(path)

        # Convert the image to RGB format (if not already)
        image = image.convert("RGB")
        rgb_array = np.array(image)
        R, G, B, RG, GB, RB, rgb_array = rgb_images(
            image, image_names[i], experiment_name
        )

        for k in k_values:
            compressed_img, variance_ratio = compress_image_rgb(
                rgb_array,
                k,
            )
            variance_ratios.append(variance_ratio)
            img_pil = Image.fromarray(compressed_img)
            save_path = os.path.join(svd_dir, f"compressed_{image_names[i]}_k{k}.jpg")
            img_pil.save(save_path, "JPEG")

    # Save variance ratios
    with open(os.path.join(experiment_name, "data.pkl"), "wb") as f:
        pickle.dump(variance_ratios, f)


def experimento_1(predictions):
    # Initialize counters
    total_images = 0
    images_with_detections = 0
    high_conf_detections = 0
    all_detections = 0
    quality_conf_map = {}

    for result in predictions:
        total_images += 1  # Count processed images
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

        # If there are detections, update counters
        if len(confs) > 0:
            images_with_detections += 1
            all_detections += len(confs)
            high_conf_detections += sum(confs >= 0.8)

        # Extract 'k' value from filename
        image_path = result.path
        filename = os.path.basename(image_path)
        match = re.search(r"_k(\d+)\.jpg$", filename)

        if match:
            k_value = int(match.group(1))

            if k_value not in quality_conf_map:
                quality_conf_map[k_value] = []

            quality_conf_map[k_value].extend(confs)

    # Compute failure percentage (images with no detections)
    failure_percentage = ((total_images - images_with_detections) / total_images) * 100

    # Compute high-confidence detection percentage
    if total_images > 0:
        high_conf_percentage = (high_conf_detections / total_images) * 100
    else:
        high_conf_percentage = 0

    print(f"Total images processed: {total_images}")
    print(
        f"Images with detections: {images_with_detections} ({100 - failure_percentage:.2f}% success rate)"
    )
    print(
        f"Images with no detections: {total_images - images_with_detections} ({failure_percentage:.2f}% failure rate)"
    )

    # Compute average confidence for each k level
    for k, confs in sorted(quality_conf_map.items()):
        avg_conf = np.mean(confs) if len(confs) > 0 else 0
        print(f"Quality level k={k}: Average confidence = {avg_conf:.3f}")


def grayscale_conversion(input_folder="ooid_general_experiment/rgb", 
                         output_folder="ooid_general_experiment/gray"):
    """
    Converts all images in input_folder to grayscale and saves them in output_folder.
    
    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where grayscale images will be saved.
    """

    # Ensure the output folder exists
    ensure_directory_exists(output_folder)

    # Ensure the input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"❌ Input folder not found: {input_folder}")

    # Process images
    converted_count = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img.save(os.path.join(output_folder, filename))  
            converted_count += 1

    print(f"✅ Grayscale conversion completed. {converted_count} images saved in {output_folder}.")