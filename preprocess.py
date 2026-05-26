from pathlib import Path
import os

import cv2

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cs131_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


IMAGE_SIZE = (512, 512)


def load_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def resize_image(image):
    return cv2.resize(image, IMAGE_SIZE)


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image):
    return cv2.GaussianBlur(gray_image, (5, 5), 0)


def detect_edges(gray_image):
    # Run Canny edge detection to find strong edges in the image.
    return cv2.Canny(gray_image, 50, 150)


def edge_overlap_score(authentic_edges, suspected_edges):
    authentic_binary = authentic_edges > 0
    suspected_binary = suspected_edges > 0

    intersection = np.logical_and(authentic_binary, suspected_binary).sum()
    union = np.logical_or(authentic_binary, suspected_binary).sum()

    if union == 0:
        return 0.0

    return intersection / union


def visualization(original, gray, blurred, output_path):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    images = [original_rgb, gray, blurred]
    titles = ["Resized Image", "Grayscale", "Gaussian Blur"]
    cmaps = [None, "gray", "gray"]

    plt.figure(figsize=(12, 4))
    for index, (image, title, cmap) in enumerate(zip(images, titles, cmaps), start=1):
        plt.subplot(1, 3, index)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def comparison_visualization(
    authentic_resized,
    suspected_resized,
    authentic_edges,
    suspected_edges,
    edge_difference,
    output_path
):
    # Convert images from BGR to RGB so matplotlib displays the colors correctly.
    authentic_rgb = cv2.cvtColor(authentic_resized, cv2.COLOR_BGR2RGB)
    suspected_rgb = cv2.cvtColor(suspected_resized, cv2.COLOR_BGR2RGB)

    # Show the original resized images, their edge maps, and the difference map.
    images = [
        authentic_rgb,
        suspected_rgb,
        authentic_edges,
        suspected_edges,
        edge_difference,
    ]

    titles = [
        "Authentic Image",
        "Suspected Image",
        "Authentic Edges",
        "Suspected Edges",
        "Edge Difference",
    ]

    cmaps = [None, None, "gray", "gray", "gray"]

    plt.figure(figsize=(14, 6))

    for index, (image, title, cmap) in enumerate(zip(images, titles, cmaps), start=1):
        plt.subplot(1, 5, index)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Run the full preprocessing pipeline on one image.
def preprocess_image(image_path, output_path):
    image = load_image(image_path)
    resized = resize_image(image)
    gray = convert_to_grayscale(resized)
    blurred = apply_gaussian_blur(gray)
    visualization(resized, gray, blurred, output_path)


# Run preprocessing and edge detection on two images without warping.
def compare_two_images(authentic_path, suspected_path, output_path):
    # Load both images.
    authentic = load_image(authentic_path)
    suspected = load_image(suspected_path)

    # Resize both images to the same size.
    authentic_resized = resize_image(authentic)
    suspected_resized = resize_image(suspected)

    # Convert both images to grayscale.
    authentic_gray = convert_to_grayscale(authentic_resized)
    suspected_gray = convert_to_grayscale(suspected_resized)

    # Blur both grayscale images before edge detection.
    authentic_blurred = apply_gaussian_blur(authentic_gray)
    suspected_blurred = apply_gaussian_blur(suspected_gray)

    # Detect edges in both images.
    authentic_edges = detect_edges(authentic_blurred)
    suspected_edges = detect_edges(suspected_blurred)

    # Compare the two edge maps directly.
    edge_difference = cv2.absdiff(authentic_edges, suspected_edges)
    score = edge_overlap_score(authentic_edges, suspected_edges)

    # Save the side-by-side comparison.
    comparison_visualization(
        authentic_resized,
        suspected_resized,
        authentic_edges,
        suspected_edges,
        edge_difference,
        output_path
    )

    return score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two shoe images with edge detection, without warping."
    )

    parser.add_argument(
        "--authentic",
        default="data/raw/authentic_airforce1/0287.jpg",
        help="Path to the authentic/reference shoe image.",
    )

    parser.add_argument(
        "--suspected",
        default="data/raw/counterfeit_airforce1/0016.jpg",
        help="Path to the suspected/counterfeit shoe image.",
    )

    parser.add_argument(
        "--output",
        default="edge_comparison_visualization.png",
        help="Path where the visualization will be saved.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score = compare_two_images(args.authentic, args.suspected, output_path)

    print(f"Edge overlap score: {score:.4f}")
    print(f"Saved edge comparison visualization to {output_path}")
