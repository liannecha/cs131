from pathlib import Path
import os
import cv2
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cs131_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import (
    convert_to_grayscale,
    edge_overlap_score,
    find_sift_matches,
    get_shoe_edges,
    load_image,
    normalize_contrast,
    resize_image,
)

# Weights for combined authenticity score
EDGE_WEIGHT = 0.45
CONTOUR_WEIGHT = 0.20
FEATURE_WEIGHT = 0.35

# Identify SIFT keypoints between the authentic and suspected shoe images
def prepare_sift_image(image_path):
    image = load_image(image_path)
    resized = resize_image(image)
    gray = convert_to_grayscale(resized)
    normalized = normalize_contrast(gray)

    return resized, normalized

# Circle red for unmatched suspected keypoints, green for matched keypoints, and draw lines between matched keypoints.
def draw_keypoint_circles(image, keypoints, keypoint_indices, color):
    output = image.copy()

    for keypoint_index in keypoint_indices:
        x, y = keypoints[keypoint_index].pt
        radius = max(3, min(8, int(keypoints[keypoint_index].size / 2)))
        cv2.circle(output, (int(x), int(y)), radius, color, 2)

    return output

# Normalized feature match score (percent of successful keypoint matches)
def feature_match_score(good_match_count, authentic_keypoint_count, suspected_keypoint_count):
    """
    Higher score = more similar local visual features
    Lower score = fewer features matched
    This could mean: 
        - the shoes are different
        - the angle is too different
        - the image quality is bad
        - preprocessing failed to extract good features
    """
    smaller_keypoint_count = min(authentic_keypoint_count, suspected_keypoint_count)

    if smaller_keypoint_count == 0:
        return 0.0

    return good_match_count / smaller_keypoint_count

# Helper for contour similarity 
def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)

# Shows shape similarity (of shoe's silhoutte); this is good for when the images are taken from the same angle
def contour_similarity_score(authentic_mask, suspected_mask):
    authentic_contour = largest_contour(authentic_mask)
    suspected_contour = largest_contour(suspected_mask)

    if authentic_contour is None or suspected_contour is None:
        return 0.0

    contour_distance = cv2.matchShapes(
        authentic_contour,
        suspected_contour,
        cv2.CONTOURS_MATCH_I1,
        0.0,
    )

    return 1.0 / (1.0 + contour_distance)

def combined_authenticity_score(edge_score, contour_score, feature_score):
    return (
        EDGE_WEIGHT * edge_score
        + CONTOUR_WEIGHT * contour_score
        + FEATURE_WEIGHT * feature_score
    )


def add_results_table(
    axis,
    authentic_keypoint_count,
    suspected_keypoint_count,
    good_match_count,
    drawn_match_count,
    drawn_unmatched_count,
    edge_score,
    contour_score,
    sift_score,
    authenticity_score,
):
    axis.axis("off")

    table_rows = [
        ["Authentic SIFT keypoints", authentic_keypoint_count, ""],
        ["Suspected SIFT keypoints", suspected_keypoint_count, ""],
        ["Good SIFT matches", good_match_count, ""],
        ["Drawn matched keypoints", drawn_match_count, ""],
        ["Drawn unmatched suspected keypoints", drawn_unmatched_count, ""],
        ["Masked edge overlap", f"{edge_score:.4f}", f"{EDGE_WEIGHT:.2f}"],
        ["Contour similarity", f"{contour_score:.4f}", f"{CONTOUR_WEIGHT:.2f}"],
        ["SIFT feature match quality", f"{sift_score:.4f}", f"{FEATURE_WEIGHT:.2f}"],
        ["Combined authenticity score", f"{authenticity_score:.4f}", "1.00"],
    ]

    table = axis.table(
        cellText=table_rows,
        colLabels=["Metric", "Value", "Weight"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.54, 0.23, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#b8b8b8")
        if row_index == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        elif row_index == len(table_rows):
            cell.set_facecolor("#e8f1ff")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f8f8f8")

def save_sift_region_analysis(
    authentic_path,
    suspected_path,
    output_path,
    max_matches=50,
    max_unmatched=100,
):
    authentic_image, authentic_normalized = prepare_sift_image(authentic_path)
    suspected_image, suspected_normalized = prepare_sift_image(suspected_path)

    authentic_keypoints, suspected_keypoints, good_matches = find_sift_matches(
        authentic_normalized,
        suspected_normalized,
    )

    strongest_matches = sorted(good_matches, key=lambda match: match.distance)[:max_matches]
    matched_authentic_indices = {match.queryIdx for match in strongest_matches}
    matched_suspected_indices = {match.trainIdx for match in strongest_matches}

    unmatched_suspected_indices = [
        index
        for index in range(len(suspected_keypoints))
        if index not in matched_suspected_indices
    ]
    unmatched_suspected_indices = sorted(
        unmatched_suspected_indices,
        key=lambda index: suspected_keypoints[index].response,
        reverse=True,
    )[:max_unmatched]

    authentic_matched = draw_keypoint_circles(
        authentic_image,
        authentic_keypoints,
        matched_authentic_indices,
        (0, 255, 0),
    )
    suspected_highlighted = draw_keypoint_circles(
        suspected_image,
        suspected_keypoints,
        unmatched_suspected_indices,
        (0, 0, 255),
    )
    suspected_highlighted = draw_keypoint_circles(
        suspected_highlighted,
        suspected_keypoints,
        matched_suspected_indices,
        (0, 255, 0),
    )

    match_lines = cv2.drawMatches(
        authentic_image,
        authentic_keypoints,
        suspected_image,
        suspected_keypoints,
        strongest_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    authentic_edges, authentic_mask, _ = get_shoe_edges(authentic_image)
    suspected_edges, suspected_mask, _ = get_shoe_edges(suspected_image)

    edge_score = edge_overlap_score(authentic_edges, suspected_edges)
    contour_score = contour_similarity_score(authentic_mask, suspected_mask)
    sift_score = feature_match_score(
        len(good_matches),
        len(authentic_keypoints),
        len(suspected_keypoints),
    )
    authenticity_score = combined_authenticity_score(
        edge_score,
        contour_score,
        sift_score,
    )

    figure_images = [
        cv2.cvtColor(authentic_matched, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(suspected_highlighted, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(match_lines, cv2.COLOR_BGR2RGB),
    ]
    titles = [
        "Authentic Matched Keypoints",
        "Suspected Matched and Unmatched Keypoints",
        "SIFT Match Lines",
    ]

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 1.35])
    fig.suptitle(
        (
            f"Authenticity score: {authenticity_score:.4f} | "
            f"edge: {edge_score:.4f}, contour: {contour_score:.4f}, "
            f"SIFT: {sift_score:.4f}"
        ),
        fontsize=12,
    )
    for index, (image, title) in enumerate(zip(figure_images, titles), start=1):
        axis = fig.add_subplot(grid[0, index - 1])
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    table_axis = fig.add_subplot(grid[1, :])
    add_results_table(
        table_axis,
        len(authentic_keypoints),
        len(suspected_keypoints),
        len(good_matches),
        len(strongest_matches),
        len(unmatched_suspected_indices),
        edge_score,
        contour_score,
        sift_score,
        authenticity_score,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path)
    plt.close(fig)

    return (
        len(authentic_keypoints),
        len(suspected_keypoints),
        len(good_matches),
        len(strongest_matches),
        len(unmatched_suspected_indices),
        edge_score,
        contour_score,
        sift_score,
        authenticity_score,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Highlight matched and unmatched SIFT keypoints for shoe comparison."
    )

    parser.add_argument(
        "--authentic",
        default="data/raw/authentic_airforce1/0287.jpg",
        help="Path to the authentic/reference shoe image.",
    )

    parser.add_argument(
        "--suspected",
        default="data/raw/counterfeit_airforce1/0016.jpg",
        help="Path to the suspected shoe image.",
    )

    parser.add_argument(
        "--output",
        default="outputs/sift_keypoint_analysis_0287_0016.png",
        help="Path where the SIFT keypoint analysis visualization will be saved.",
    )

    parser.add_argument(
        "--max-matches",
        type=int,
        default=50,
        help="Maximum number of strongest SIFT matches to draw.",
    )

    parser.add_argument(
        "--max-unmatched",
        type=int,
        default=100,
        help="Maximum number of strongest unmatched suspected keypoints to draw.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        authentic_keypoint_count,
        suspected_keypoint_count,
        good_match_count,
        drawn_match_count,
        drawn_unmatched_count,
        edge_score,
        contour_score,
        sift_score,
        authenticity_score,
    ) = save_sift_region_analysis(
        args.authentic,
        args.suspected,
        output_path,
        args.max_matches,
        args.max_unmatched,
    )

    print(f"Authentic SIFT keypoints: {authentic_keypoint_count}")
    print(f"Suspected SIFT keypoints: {suspected_keypoint_count}")
    print(f"Good SIFT matches: {good_match_count}")
    print(f"Drawn matched keypoints: {drawn_match_count}")
    print(f"Drawn unmatched suspected keypoints: {drawn_unmatched_count}")
    print(f"Edge overlap score: {edge_score:.4f}")
    print(f"Contour similarity score: {contour_score:.4f}")
    print(f"SIFT feature match score: {sift_score:.4f}")
    print(f"Combined authenticity score: {authenticity_score:.4f}")
    print(f"Saved SIFT keypoint analysis to {output_path}")
