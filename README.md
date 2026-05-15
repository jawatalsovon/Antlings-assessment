# Drone Human & Car Detection + Counting System
### Antlings Internship Program — AI/ML Technical Assessment

A computer vision pipeline for analyzing drone/aerial imagery to detect humans and cars, count total humans, and visualize results — built on the VisDrone dataset using YOLOv8 and ByteTrack.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.2026 |
| mAP@0.5:0.95 | 0.1212 |
| Precision | 0.5356 |
| Recall | 0.2489 |
| Inference Speed | ~2.3ms per image |

> Note: VisDrone is one of the hardest aerial detection benchmarks. State-of-the-art models achieve ~35–40% mAP@0.5. A 50-epoch YOLOv8n baseline achieving 20.3% is expected and competitive for this scale of training.

---

## Demo

| Detection & Counting | Tracking |
|----------------------|----------|
| ![Detection Grid](outputs/task03_detection_grid.png) | ByteTrack tracking with trail lines and persistent IDs |

---

## Task Breakdown

### Task 01 — Dataset Understanding & Preprocessing

**Dataset:** VisDrone2019-DET — 10,209 drone images across 10 object classes captured at varying altitudes and viewpoints.

**Structure:**
- `VisDrone2019-DET-train/` — 6,471 images
- `VisDrone2019-DET-val/` — 548 images
- `VisDrone2019-DET-test-dev/` — 1,610 images

**Classes used:**
- Human = class 0 (pedestrian) + class 1 (people)
- Car = class 3

The dataset was already pre-converted to YOLO format (normalized bounding box coordinates). No manual conversion was required.

**Key Challenges:**
1. **Small objects** — humans appear as 10–30px blobs from altitude; over 60% of human annotations have area < 0.1% of the image
2. **High density** — 100+ humans per frame causes heavy occlusion and NMS suppression issues
3. **Class imbalance** — pedestrian class dominates heavily over others
4. **Scale variation** — same object appears at vastly different sizes across images due to varying drone altitude
5. **Viewpoint diversity** — top-down and oblique angles both present, requiring augmentation coverage

---

### Task 02 — Model Training

**Model:** YOLOv8n (nano) fine-tuned from COCO pretrained weights

**Why YOLOv8?**
- Real-time capable single-stage detector
- Excellent small-object detection with mosaic augmentation
- Ultralytics library provides clean training, inference, and tracking APIs

**Training Configuration:**
```
Epochs:        50 (early stopping patience=15)
Image size:    640×640
Batch size:    16
Optimizer:     AdamW (lr=0.001, weight_decay=0.0005)
Hardware:      AMD MI300X GPU (192GB VRAM) via AMD Developer Cloud
```

**Key Augmentations for Aerial Imagery:**
| Augmentation | Value | Reason |
|-------------|-------|--------|
| Mosaic | 1.0 | Combines 4 images — increases density and scale variety |
| Copy-paste | 0.1 | Redistributes rare classes |
| Vertical flip | 0.3 | Makes sense for aerial view unlike ground cameras |
| Rotation | 10° | Covers angled drone captures |
| Scale | 0.5 | Simulates different altitudes |

**Training Strategy:** Trained on all 10 VisDrone classes for richer feature learning, then filtered to human/car at inference time only.

---

### Task 03 — Human & Car Detection with Counting

The detection pipeline:
1. Image fed to trained YOLOv8n model
2. Predictions filtered by confidence threshold (0.25) and NMS (IoU=0.45)
3. Results filtered to human (class 0+1) and car (class 3)
4. Human count and car count computed
5. Bounding boxes drawn with OpenCV — green for humans, red for cars
6. Count overlay panel displayed on image

**Sample Output:**

![Detection Results](outputs/task03_detection_grid.png)

---

### Task 04 — Object Tracking with ByteTrack *(Bonus)*

Implemented ByteTrack via Ultralytics' built-in tracker:
- Assigns persistent track IDs to each detected object across frames
- Uses Kalman filtering to predict object positions between frames
- Draws movement trail lines showing trajectory history
- Live per-frame human and car count overlay

**Why ByteTrack over DeepSORT?**
ByteTrack associates ALL detections (not just high-confidence ones) with existing tracks, making it more robust in dense aerial scenes without needing a re-identification model.

---

### Task 05 — Evaluation & Visualization

![Metrics Dashboard](outputs/task05_metrics.png)

**Evaluation run on:** VisDrone2019-DET-val (548 images, 38,759 object annotations)

**Strengths:**
- Fast inference (~2.3ms/image on MI300X) — viable for real-time drone deployment
- Dual human class merging gives more complete human counts
- Mosaic augmentation significantly helps small object recall

**Limitations:**
- YOLOv8n is the smallest variant — larger models (yolov8m/l) would push mAP significantly higher
- 50 epochs is a baseline — SOTA VisDrone results require 300+ epochs
- Frame-by-frame counting without track ID persistence may double-count in video

**Improvements for production:**
- SAHI (Slicing Aided Hyper Inference) for better small object detection
- Higher input resolution (1280px)
- Track-ID-based unique person counting

---

## Repository Structure

```
├── notebook/
│   └── CV.ipynb                  # Development notebook (full workflow)
├── antlings_visdrone_v2.py        # Clean reference code
├── outputs/
│   ├── task03_detection_grid.png  # Detection results grid
│   ├── task03_result_1-6.jpg      # Individual detection outputs
│   ├── task04_tracked.mp4         # ByteTrack tracking video
│   ├── task05_metrics.png         # Evaluation metrics dashboard
│   └── counts.csv                 # Batch inference human/car counts
├── runs/
│   └── detect/                    # Training artifacts (weights, curves)
└── README.md
```

---

## Setup & Reproduction

```bash
# Install dependencies
pip install ultralytics kagglehub

# Download dataset
python -c "import kagglehub; kagglehub.dataset_download('banuprasadb/visdrone-dataset')"

# Run inference with trained weights
from ultralytics import YOLO
model = YOLO('runs/detect/weights/best.pt')
results = model.predict('your_image.jpg', conf=0.25)
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| YOLOv8 (Ultralytics) | Object detection & tracking |
| PyTorch + ROCm 7.0 | Deep learning framework on AMD GPU |
| OpenCV | Image processing & visualization |
| ByteTrack | Multi-object tracking |
| AMD MI300X | Training hardware (192GB VRAM) |
| VisDrone2019-DET | Aerial drone imagery dataset |

---

## Author

**Jawat Al Sovon**
Freshman, Institute of Business Administration (IBA), University of Dhaka
[LinkedIn](https://linkedin.com/in/jawat-al-sovon) | jawatalsovon@gmail.com
