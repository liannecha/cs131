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

# Take a grayscale image, improve the local contrast, and return the improved version.
def normalize_contrast(gray_image):
    """
    Use CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Increase clipLimit for more contrast, but more noise/sharpness.
    High clipLimit is good for low lighting and hard to see edges.
    Smaller tile size is good for more details, but introduces more noise.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def apply_gaussian_blur(gray_image):
    return cv2.GaussianBlur(gray_image, (5, 5), 0)

# We need to mask because the background has strong edges that creates too much noise.
def create_shoe_mask(image):
    height, width = image.shape[:2]

    # GrabCut is an OpenCV algorithm that separates foreground from background based on color and contrast.
    grabcut_mask = np.zeros((height, width), np.uint8)

    rectangle = (
        int(0.05 * width),
        int(0.05 * height),
        int(0.90 * width),
        int(0.90 * height),
    )

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        grabcut_mask,
        rectangle,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_RECT,
    )

    mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype("uint8")

    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    return mask

# Run Canny edge detection to find strong edges in the image.
def detect_edges(gray_image):
    return cv2.Canny(gray_image, 50, 150)

def get_shoe_edges(image):
    mask = create_shoe_mask(image)
    gray = convert_to_grayscale(image)
    normalized_gray = normalize_contrast(gray)
    blurred = apply_gaussian_blur(normalized_gray)
    masked_gray = cv2.bitwise_and(blurred, blurred, mask=mask)
    edges = detect_edges(masked_gray)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    return edges, mask, blurred

def find_sift_matches(authentic_gray, suspected_gray):
    sift = cv2.SIFT_create()

    authentic_keypoints, authentic_descriptors = sift.detectAndCompute(authentic_gray, None)
    suspected_keypoints, suspected_descriptors = sift.detectAndCompute(suspected_gray, None)

    if authentic_descriptors is None or suspected_descriptors is None:
        return authentic_keypoints, suspected_keypoints, []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(authentic_descriptors, suspected_descriptors, k=2)

    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) != 2:
            continue

        best_match, second_best_match = match_pair
        if best_match.distance < 0.75 * second_best_match.distance:
            good_matches.append(best_match)

    return authentic_keypoints, suspected_keypoints, good_matches


def save_keypoint_match_visualization(
    authentic_image,
    suspected_image,
    authentic_keypoints,
    suspected_keypoints,
    matches,
    output_path
):
    sorted_matches = sorted(matches, key=lambda match: match.distance)
    matches_to_draw = sorted_matches[:50]

    match_visualization = cv2.drawMatches(
        authentic_image,
        authentic_keypoints,
        suspected_image,
        suspected_keypoints,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(str(output_path), match_visualization)


def edge_overlap_score(authentic_edges, suspected_edges):
    authentic_binary = authentic_edges > 0
    suspected_binary = suspected_edges > 0

    intersection = np.logical_and(authentic_binary, suspected_binary).sum()
    union = np.logical_or(authentic_binary, suspected_binary).sum()

    if union == 0:
        return 0.0

    return intersection / union


def visualization(original, gray, normalized, blurred, output_path):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    images = [original_rgb, gray, normalized, blurred]
    titles = ["Resized Image", "Grayscale", "Contrast Normalized", "Gaussian Blur"]
    cmaps = [None, "gray", "gray", "gray"]

    plt.figure(figsize=(14, 4))
    for index, (image, title, cmap) in enumerate(zip(images, titles, cmaps), start=1):
        plt.subplot(1, 4, index)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def comparison_visualization(
    authentic_resized,
    suspected_resized,
    authentic_mask,
    suspected_mask,
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
        authentic_mask,
        suspected_mask,
        authentic_edges,
        suspected_edges,
        edge_difference,
    ]

    titles = [
        "Authentic Image",
        "Suspected Image",
        "Authentic Mask",
        "Suspected Mask",
        "Authentic Edges",
        "Suspected Edges",
        "Edge Difference",
    ]

    cmaps = [None, None, "gray", "gray", "gray", "gray", "gray"]

    plt.figure(figsize=(18, 6))

    for index, (image, title, cmap) in enumerate(zip(images, titles, cmaps), start=1):
        plt.subplot(1, 7, index)
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
    normalized = normalize_contrast(gray)
    blurred = apply_gaussian_blur(normalized)
    visualization(resized, gray, normalized, blurred, output_path)


# Run preprocessing, SIFT matching, and edge detection on two images.
def compare_two_images(authentic_path, suspected_path, output_path, matches_output_path=None):
    # Load both images.
    authentic = load_image(authentic_path)
    suspected = load_image(suspected_path)

    # Resize both images to the same size.
    authentic_resized = resize_image(authentic)
    suspected_resized = resize_image(suspected)

    # Convert both images to grayscale.
    authentic_gray = convert_to_grayscale(authentic_resized)
    suspected_gray = convert_to_grayscale(suspected_resized)

    # Normalize contrast before blurring so lighting differences affect edges less.
    authentic_normalized = normalize_contrast(authentic_gray)
    suspected_normalized = normalize_contrast(suspected_gray)

    # Detect SIFT keypoints and match distinctive regions between images.
    authentic_keypoints, suspected_keypoints, good_matches = find_sift_matches(
        authentic_normalized,
        suspected_normalized,
    )

    # Mask out the background before detecting shoe edges.
    authentic_edges, authentic_mask, _ = get_shoe_edges(authentic_resized)
    suspected_edges, suspected_mask, _ = get_shoe_edges(suspected_resized)

    # Compare the two edge maps directly.
    edge_difference = cv2.absdiff(authentic_edges, suspected_edges)
    score = edge_overlap_score(authentic_edges, suspected_edges)

    # Save the side-by-side comparison.
    comparison_visualization(
        authentic_resized,
        suspected_resized,
        authentic_mask,
        suspected_mask,
        authentic_edges,
        suspected_edges,
        edge_difference,
        output_path
    )

    if matches_output_path is not None:
        save_keypoint_match_visualization(
            authentic_resized,
            suspected_resized,
            authentic_keypoints,
            suspected_keypoints,
            good_matches,
            matches_output_path,
        )

    return (
        score,
        len(authentic_keypoints),
        len(suspected_keypoints),
        len(good_matches),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two shoe images with SIFT matching and edge detection."
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

    parser.add_argument(
        "--matches-output",
        default=None,
        help="Optional path where the SIFT keypoint match visualization will be saved.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches_output_path = None
    if args.matches_output is not None:
        matches_output_path = Path(args.matches_output)
        matches_output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        score,
        authentic_keypoint_count,
        suspected_keypoint_count,
        match_count,
    ) = compare_two_images(
        args.authentic,
        args.suspected,
        output_path,
        matches_output_path,
    )

    print(f"Edge overlap score: {score:.4f}")
    print(f"Authentic SIFT keypoints: {authentic_keypoint_count}")
    print(f"Suspected SIFT keypoints: {suspected_keypoint_count}")
    print(f"Good SIFT matches: {match_count}")
    print(f"Saved edge comparison visualization to {output_path}")
    if matches_output_path is not None:
        print(f"Saved SIFT match visualization to {matches_output_path}")
