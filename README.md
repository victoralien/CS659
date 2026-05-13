# Counting Coins: Classical CV vs. Deep Learning

**CS659 – Visual Perception and Learning | San Diego State University**

A comparison of two approaches to U.S. coin detection and denomination classification from a single image: a classical computer vision pipeline using Hough circles and geometric reasoning, and a YOLO-based deep learning detector.

## Overview

This project builds and evaluates two complete coin-counting systems:

1. **Classical pipeline** — grayscale conversion, CLAHE contrast enhancement, Gaussian smoothing, Hough Circle Transform, scale estimation, and rule-based denomination assignment using known physical coin diameters.

2. **Deep learning pipeline** — a YOLO-based object detector fine-tuned on a labeled U.S. coins dataset, producing bounding boxes and class labels directly from the input image.

The classical approach is lightweight and interpretable, but denomination accuracy degrades quickly under real imaging conditions such as shadows, camera angle, and overlap. The YOLO model is more robust, achieving a peak validation F1 score of **0.9222** with precision **0.9562** and recall **0.8905** at a confidence threshold of **0.40** and IoU threshold of **0.50**.

## File Manifest

| File | Description |
|---|---|
| `coin_counter.py` | Interactive Tkinter desktop app for the classical CV pipeline. Upload an image, tune Hough parameters with sliders, and view annotated output with a coin-value breakdown. |
| `best.pt` | Trained YOLO model weights, using the best checkpoint from training. Used for inference in the model testing notebook. |
| `CS659_Project.pdf` | Final written report. Covers methodology, experiments, threshold grid search results, confusion matrix analysis, and conclusions. |
| `CS659_CoinCounterTraining.ipynb` | Training notebook. Loads the Roboflow U.S. Coins Dataset, configures and trains the YOLO detector, and saves weights. |
| `CS659_CoinCounter_Model_Testing.ipynb` | Evaluation notebook. Loads `best.pt`, runs the threshold grid search over confidence and IoU combinations, and produces validation metrics and diagnostic plots. |
| `CS659_DeepLearningUses.ipynb` | Exploratory notebook covering broader deep learning background and use cases relevant to the project context. |

## Dataset

The deep learning model was trained and evaluated on the **U.S. Coins Dataset** by A. Tatham, available on Roboflow Universe:

[https://universe.roboflow.com/atathamuscoinsdataset/u.s.-coins-dataset-a.tatham](https://universe.roboflow.com/atathamuscoinsdataset/u.s.-coins-dataset-a.tatham)

The dataset contains labeled RGB images with bounding boxes for four classes: dime, nickel, penny, and quarter, exported in YOLO format.

## Requirements

### Classical pipeline (`coin_counter.py`)

- Python >= 3.9
- opencv-python
- Pillow
- numpy
- tkinter
  - Included in standard Python on most platforms

Install dependencies:

```bash
pip install opencv-python Pillow numpy
```

### Deep learning notebooks

- Python >= 3.9
- ultralytics
- torch
- torchvision
- numpy
- matplotlib
- Pillow

Install dependencies:

```bash
pip install ultralytics torch torchvision matplotlib Pillow
```

**GPU note:** Training was done with CUDA. CPU inference works but is slower. If you have an NVIDIA GPU, make sure you have a compatible CUDA-enabled PyTorch build installed before running the training notebook.

## Usage

### Classical CV Desktop App

Run:

```bash
python coin_counter.py
```

1. Click **Upload Image** and select a `.png`, `.jpg`, `.jpeg`, or `.bmp` photo of U.S. coins.
2. Click **Analyze** to run detection.
3. Use the parameter sliders on the left to tune Hough circle detection for your image.

Hough parameters:

| Parameter | Description |
|---|---|
| `dp` | Inverse resolution ratio for the accumulator. |
| `minDist` | Minimum distance between detected circle centers. |
| `param1` | Upper Canny edge threshold. |
| `param2` | Hough accumulator threshold. Lower values produce more detections but more false positives. |
| `minRadius` / `maxRadius` | Expected coin radius range in pixels. |

The annotated image, per-denomination breakdown, and estimated total value are displayed immediately.

### Tips for Best Classical Pipeline Results

- Use a plain, high-contrast background such as white paper or cardboard.
- Avoid overlapping coins; spacing them out improves circle detection.
- If coins are missed, lower `param2`.
- If false circles appear, raise `param2`.
- Adjust `minRadius` and `maxRadius` to match your image resolution.

## YOLO Inference

Open `CS659_CoinCounter_Model_Testing.ipynb` in Jupyter or Google Colab and run all cells.

The notebook:

- Loads `best.pt` with the Ultralytics YOLO API.
- Runs the grid search over confidence values `{0.25, 0.35, 0.40, 0.50, 0.60}` and IoU values `{0.40, 0.50, 0.60, 0.70}`.
- Reports precision, recall, and F1 for each combination.
- Generates the recall-confidence curve, F1-confidence curve, and normalized confusion matrix.

To run inference on a custom image:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("your_image.jpg", conf=0.40, iou=0.50)
results[0].show()
```

## Training from Scratch

Open `CS659_CoinCounterTraining.ipynb`.

You will need:

- The [Roboflow U.S. Coins Dataset](https://universe.roboflow.com/atathamuscoinsdataset/u.s.-coins-dataset-a.tatham) exported in YOLO format.
- A `data.yaml` file pointing to your train and validation image directories.

Update the dataset path in the notebook, then run all cells. Training uses Ultralytics YOLO with pretrained weights and an image size of `640x640`.

## Results Summary

| Metric | Value |
|---|---:|
| Best confidence threshold | 0.40 |
| Best IoU threshold | 0.50 |
| Validation precision | 0.9562 |
| Validation recall | 0.8905 |
| Validation F1 | 0.9222 |
| mAP@0.5 | 0.8997 |
| mAP@0.5:0.95 | 0.7013 |

Per-class performance from the confusion matrix diagonal:

| Class | Approximate Score |
|---|---:|
| Penny | 0.96 |
| Quarter | 0.91 |
| Nickel | 0.84 |
| Dime | 0.83 |

See `CS659_Project.pdf` for full results, figures, and analysis.

## Project Structure

```text
.
├── coin_counter.py                        # Classical CV desktop app
├── best.pt                                # Trained YOLO weights
├── CS659_Project.pdf                      # Final report
├── CS659_CoinCounterTraining.ipynb        # YOLO training notebook
├── CS659_CoinCounter_Model_Testing.ipynb  # Evaluation & grid search notebook
├── CS659_DeepLearningUses.ipynb           # Deep learning background notebook
└── README.md
```

## Key Findings

The classical pipeline reliably locates coins as circular objects, but denomination classification via apparent radius is fragile. Shadows, slight camera angle changes, and imperfect circle boundaries all shift the measured radius enough to cause misclassification, particularly between visually similar pairs like nickels and quarters.

The YOLO detector learns richer appearance cues, including texture, color, and context, and is far more stable across real image conditions. It successfully handles clutter, lighting variation, and partial overlap without any hand-tuned geometric rules. For a task that requires both localization and fine-grained classification, the learned detector is the stronger end-to-end solution.

## References

- Duda & Hart (1972) — Hough Transform for line and circle detection.
- Canny (1986) — Computational approach to edge detection.
- Lowe (2004) — SIFT features.
- Dalal & Triggs (2005) — Histograms of Oriented Gradients.
- Krizhevsky, Sutskever & Hinton (2012) — AlexNet / deep CNNs.
- Redmon et al. (2016) — You Only Look Once (YOLO).

## Course

**CS659 – Visual Perception and Learning**  
San Diego State University














