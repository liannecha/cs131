# Nike Shoe Authenticity Comparison

This project is an early computer vision prototype for comparing authentic and suspected Nike shoe images.

The current program compares two images by resizing them, converting them to grayscale, normalizing contrast with CLAHE, detecting and matching SIFT keypoints, creating a foreground shoe mask with GrabCut, keeping the largest connected foreground component, applying Canny edge detection only inside the masked shoe area, computing an edge overlap score, and saving visualizations.

## Dataset

Raw images are stored in:

- `data/raw/authentic_airforce1/`
- `data/raw/counterfeit_airforce1/`
- `data/raw/authentic_jordan1/`
- `data/raw/counterfeit_jordan1/`

## Run

Run the current comparison script from the project root:

```bash
.venv/bin/python preprocess.py \
  --authentic data/raw/authentic_airforce1/0287.jpg \
  --suspected data/raw/counterfeit_airforce1/0016.jpg \
  --output outputs/grabcut_masked_edge_comparison_0287_0016.png \
  --matches-output outputs/sift_matches_0287_0016.png
```

The script prints an edge overlap score, the number of SIFT keypoints found in each image, and the number of good SIFT matches. It saves the edge comparison image to the path given by `--output` and the keypoint match image to the path given by `--matches-output`.

## Current Output

The saved visualization shows:

- the authentic image
- the suspected image
- the authentic shoe mask
- the suspected shoe mask
- the authentic edge map
- the suspected edge map
- the edge difference map

The edge overlap score is a rough similarity score between the masked edge maps. A higher score means more detected shoe edges overlap between the authentic and suspected images.

The SIFT match visualization draws lines between distinctive keypoints that appear similar in both images.

## SIFT Keypoint Analysis

Run this script to highlight matched and unmatched SIFT keypoints:

```bash
.venv/bin/python analysis.py \
  --authentic data/raw/authentic_airforce1/0287.jpg \
  --suspected data/raw/counterfeit_airforce1/0016.jpg \
  --output outputs/sift_keypoint_analysis_0287_0016.png
```

The analysis image uses green circles for matched keypoints and red circles for strong suspected-image keypoints that did not appear in the strongest SIFT matches. It also includes a results table with metrics, values, weights, and the rough combined authenticity score.

The combined score is a prototype similarity score, not a true probability. It is currently weighted as:

- 45% masked edge overlap
- 20% contour similarity
- 35% SIFT feature-match quality

## Current Limitations

The shoe mask is still a rough foreground segmentation, so it can struggle if GrabCut includes the wrong foreground object or if an image contains multiple shoes. The program also does not align or warp images, so the edge overlap score can be unreliable when the shoes are positioned differently.
