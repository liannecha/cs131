from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


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

# Run the full preprocessing pipeline on one image.
def preprocess_image(image_path, output_path):
    image = load_image(image_path)
    resized = resize_image(image)
    gray = convert_to_grayscale(resized)
    blurred = apply_gaussian_blur(gray)
    visualization(resized, gray, blurred, output_path)


if __name__ == "__main__":
    
    # Allows us to choose the input image and output file from the terminal.
    # Provides default paths if no arguments are given.
    import argparse

    parser = argparse.ArgumentParser(description="Simple shoe image preprocessing demo.")
    parser.add_argument(
        "--image",
        default="data/raw/authentic_airforce1/2455.jpg",
        help="Path to an input image.",
    )
    parser.add_argument(
        "--output",
        default="preprocess_visualization.png",
        help="Path where the side-by-side visualization will be saved.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocess_image(args.image, output_path)
    print(f"Saved preprocessing visualization to {output_path}")
