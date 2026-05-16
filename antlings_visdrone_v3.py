# %% [markdown]
# # Antlings Internship – AI/ML Technical Assessment
# ## Drone Human & Car Detection + Counting System
# ### VisDrone Dataset | YOLOv8 | ByteTrack
#
# **Tasks covered:**
# - Task 01: Dataset Understanding & Preprocessing
# - Task 02: Model Training (YOLOv8)
# - Task 03: Human & Car Detection with Counting
# - Task 04: Object Tracking with ByteTrack (Bonus)
# - Task 05: Evaluation & Visualization

# %% [markdown]
# ## Cell 1 — Environment Check

# %%
import subprocess, sys

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# Check GPU
import torch
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA/ROCm:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected — training will be slow on CPU")

# %% [markdown]
# ## Cell 2 — Install Dependencies

# %%
run("pip install -q ultralytics kaggle opencv-python-headless matplotlib seaborn pandas tqdm Pillow")

# Verify ultralytics
from ultralytics import YOLO
import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

# %% [markdown]
# ## Cell 3 — Download VisDrone Dataset from Kaggle
#
# **Before running:** Upload your `kaggle.json` API token to this environment.
# Get it from: kaggle.com → Account → API → Create New Token

# %%
import os

# ---- Option A: Upload kaggle.json manually then run this ----
# from google.colab import files      # (Colab only)
# files.upload()                      # upload kaggle.json

# Setup kaggle credentials
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# If kaggle.json is in current dir, move it
if os.path.exists("kaggle.json"):
    run(f"cp kaggle.json {kaggle_dir}/kaggle.json")
    run(f"chmod 600 {kaggle_dir}/kaggle.json")
    print("kaggle.json configured.")
else:
    print("ACTION NEEDED: Place kaggle.json in current directory and re-run.")
    print("Or set env vars: KAGGLE_USERNAME and KAGGLE_KEY")

# %% [markdown]
# ### Download & Extract

# %%
DATASET_DIR = "./visdrone_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

if not os.path.exists(f"{DATASET_DIR}/VisDrone2019-DET-train"):
    print("Downloading VisDrone dataset (~2 GB)...")
    run(f"kaggle datasets download -d banuprasadb/visdrone-dataset -p {DATASET_DIR} --unzip")
    print("Download complete.")
else:
    print("Dataset already exists, skipping download.")

# Inspect extracted structure
for item in sorted(os.listdir(DATASET_DIR)):
    print(f"  {item}/")

# %% [markdown]
# ## Task 01 — Dataset Understanding & Preprocessing

# %% [markdown]
# ### Cell 4 — Understand Dataset Structure

# %%
import glob
import pandas as pd
import numpy as np

# VisDrone annotation format (each line):
# bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
#
# Categories:
# 0=ignored  1=pedestrian  2=people  3=bicycle  4=car  5=van
# 6=truck    7=tricycle    8=awning-tricycle     9=bus  10=motor  11=others

VISDRONE_CATEGORIES = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle",
    8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others"
}

# For our task: humans = pedestrian(1) + people(2), car = car(4)
TARGET_MAP = {1: 0, 2: 0, 4: 1}  # 0=human, 1=car
CLASS_NAMES = ["human", "car"]

# Locate annotation files
train_ann_dir = glob.glob(f"{DATASET_DIR}/**/annotations*", recursive=True)
print("Annotation directories found:")
for d in train_ann_dir:
    print(f"  {d}")

# %%
# Parse all annotations for statistics
def parse_visdrone_ann(ann_path):
    rows = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h, score, cat = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            rows.append({"x": x, "y": y, "w": w, "h": h, "score": score, "category": cat,
                         "category_name": VISDRONE_CATEGORIES.get(cat, "unknown")})
    return rows

# Collect stats from train set
ann_files = sorted(glob.glob(f"{DATASET_DIR}/**/annotations/*.txt", recursive=True))
print(f"\nTotal annotation files found: {len(ann_files)}")

# Sample stats from first 500 files
all_rows = []
for af in ann_files[:500]:
    all_rows.extend(parse_visdrone_ann(af))

df = pd.DataFrame(all_rows)
print(f"\nTotal objects in sample (500 images): {len(df)}")
print("\nObject category distribution:")
print(df["category_name"].value_counts())

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Class distribution bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# All categories
cat_counts = df["category_name"].value_counts()
axes[0].bar(cat_counts.index, cat_counts.values, color="#2563EB", edgecolor="white")
axes[0].set_title("All Object Categories (VisDrone)", fontsize=13)
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)

# Target classes only
target_df = df[df["category"].isin([1, 2, 4])]
target_df["class"] = target_df["category"].map({1: "human(pedestrian)", 2: "human(people)", 4: "car"})
target_counts = target_df["class"].value_counts()
axes[1].bar(target_counts.index, target_counts.values, color=["#16A34A", "#15803D", "#DC2626"])
axes[1].set_title("Target Classes for Our Task", fontsize=13)
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("task01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: task01_class_distribution.png")

# %%
# Bounding box size analysis
target_df["area"] = target_df["w"] * target_df["h"]
print("\nBounding Box Statistics (target classes):")
print(target_df.groupby("class")[["w", "h", "area"]].describe().round(1))

# Area histogram
fig, ax = plt.subplots(figsize=(10, 4))
for cls, color in [("human(pedestrian)", "#16A34A"), ("car", "#DC2626")]:
    subset = target_df[target_df["class"] == cls]["area"]
    ax.hist(subset.clip(upper=5000), bins=60, alpha=0.7, label=cls, color=color)
ax.set_title("Bounding Box Area Distribution (clipped at 5000px²)", fontsize=12)
ax.set_xlabel("Area (pixels²)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("task01_bbox_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Cell 5 — Sample Image Visualization with Annotations

# %%
def visualize_sample(img_path, ann_path, max_objects=200, figsize=(14, 8)):
    """Draw bounding boxes on a VisDrone image."""
    img = np.array(Image.open(img_path).convert("RGB"))
    anns = parse_visdrone_ann(ann_path)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    color_map = {1: "#22C55E", 2: "#86EFAC", 4: "#EF4444", 0: "#94A3B8"}  # ignored = gray

    for ann in anns[:max_objects]:
        cat = ann["category"]
        if cat == 0:
            continue
        color = color_map.get(cat, "#F59E0B")
        rect = patches.Rectangle((ann["x"], ann["y"]), ann["w"], ann["h"],
                                   linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color="#22C55E", linewidth=2, label="pedestrian"),
        Line2D([0], [0], color="#86EFAC", linewidth=2, label="people"),
        Line2D([0], [0], color="#EF4444", linewidth=2, label="car"),
        Line2D([0], [0], color="#F59E0B", linewidth=2, label="other"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)
    ax.set_title(f"Sample: {os.path.basename(img_path)}", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    return fig

# Find sample images
img_files = sorted(glob.glob(f"{DATASET_DIR}/**/images/*.jpg", recursive=True))
print(f"Total images found: {len(img_files)}")

# Show 3 samples
import random
random.seed(42)
samples = random.sample(img_files[:500], min(3, len(img_files)))

for img_path in samples:
    # Derive annotation path
    ann_path = img_path.replace("images", "annotations").replace(".jpg", ".txt")
    if os.path.exists(ann_path):
        fig = visualize_sample(img_path, ann_path)
        fname = f"task01_sample_{os.path.basename(img_path).replace('.jpg','')}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.show()
        print(f"Saved: {fname}")

# %% [markdown]
# ### Cell 6 — Convert VisDrone → YOLO Format

# %%
from pathlib import Path
from tqdm import tqdm

YOLO_DIR = Path("./visdrone_yolo")

def convert_split(split_name, img_src_dir, ann_src_dir):
    """Convert one split (train/val/test) to YOLO format."""
    out_img = YOLO_DIR / split_name / "images"
    out_lbl = YOLO_DIR / split_name / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(Path(img_src_dir).glob("*.jpg"))
    skipped = 0

    for img_path in tqdm(img_paths, desc=f"Converting {split_name}"):
        ann_path = Path(ann_src_dir) / (img_path.stem + ".txt")
        if not ann_path.exists():
            skipped += 1
            continue

        # Get image dimensions
        with Image.open(img_path) as im:
            W, H = im.size

        yolo_lines = []
        for line in open(ann_path):
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h, score, cat = (int(p) for p in parts[:6])

            # Skip ignored regions and unwanted categories
            if cat not in TARGET_MAP:
                continue
            if w <= 0 or h <= 0:
                continue

            cls_id = TARGET_MAP[cat]
            # Convert to YOLO normalized (x_center, y_center, width, height)
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            wn = w / W
            hn = h / H
            # Clamp to [0, 1]
            xc, yc, wn, hn = (max(0, min(1, v)) for v in [xc, yc, wn, hn])
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        # Symlink or copy image
        dst_img = out_img / img_path.name
        if not dst_img.exists():
            import shutil
            shutil.copy2(img_path, dst_img)

        # Write label file
        with open(out_lbl / (img_path.stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"  {split_name}: {len(img_paths) - skipped} images converted, {skipped} skipped")

# Detect split directories
splits_info = {}
for split in ["train", "val", "test"]:
    img_dirs = glob.glob(f"{DATASET_DIR}/**/*{split}*/images", recursive=True)
    ann_dirs = glob.glob(f"{DATASET_DIR}/**/*{split}*/annotations", recursive=True)
    if img_dirs and ann_dirs:
        splits_info[split] = (img_dirs[0], ann_dirs[0])
        print(f"Found {split}: {img_dirs[0]}")

for split, (img_d, ann_d) in splits_info.items():
    convert_split(split, img_d, ann_d)

print("\nConversion complete.")

# %%
# Write data.yaml for YOLOv8
yaml_content = f"""
path: {YOLO_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: 2
names: ['human', 'car']
"""

yaml_path = YOLO_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content.strip())

print(f"data.yaml written to: {yaml_path}")
print(yaml_content)

# Verify label counts
for split in splits_info:
    n_lbl = len(list((YOLO_DIR / split / "labels").glob("*.txt")))
    n_img = len(list((YOLO_DIR / split / "images").glob("*.jpg")))
    print(f"{split:5s}: {n_img} images, {n_lbl} labels")

# %% [markdown]
# ### Cell 7 — Dataset Challenges Summary (Task 01)

# %%
print("""
======================================================
DATASET CHALLENGES (VisDrone)
======================================================
1. SMALL OBJECTS: Humans appear as tiny 5-20px blobs
   from aerial view — standard anchors miss them.

2. HIGH DENSITY: 100+ people can appear in one frame,
   causing heavy overlap and occlusion.

3. CLASS IMBALANCE: pedestrian >> car >> others.
   Risk: model biased toward dominant class.

4. VARYING ALTITUDE: Drone heights differ across scenes,
   causing large scale variation in object sizes.

5. VIEWPOINT: Top-down vs oblique angles give different
   object appearances — harder to generalise.

6. LIGHTING/WEATHER: Some images are low-light, foggy,
   or have strong shadows.

7. IGNORED REGIONS: Category 0 must be excluded from
   training to avoid corrupting gradients.
======================================================
""")

# %% [markdown]
# ## Task 02 — Model Training

# %% [markdown]
# ### Cell 8 — Train YOLOv8 on VisDrone

# %%
import torch
from ultralytics import YOLO

# Use YOLOv8n (nano) for speed; swap to yolov8s/m for better accuracy
MODEL_NAME = "yolov8n.pt"

model = YOLO(MODEL_NAME)
print(f"Loaded base model: {MODEL_NAME}")
print(f"Training on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# %%
# Training — adjust epochs/batch based on available time
results = model.train(
    data=str(yaml_path.resolve()),
    epochs=50,           # 50 good for assessment; 100+ for production
    imgsz=640,           # standard; try 1280 if VRAM allows
    batch=16,            # reduce to 8 if OOM
    device=0 if torch.cuda.is_available() else "cpu",
    workers=4,
    patience=15,         # early stopping
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,          # mosaic augmentation (great for small objects)
    mixup=0.1,
    copy_paste=0.1,
    flipud=0.3,          # vertical flip (aerial view benefits from this)
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,        # small rotation for aerial robustness
    translate=0.1,
    scale=0.5,
    name="visdrone_human_car",
    project="antlings_runs",
    exist_ok=True,
    verbose=True,
)

print("\nTraining complete!")
print(f"Best weights: {results.save_dir}/weights/best.pt")

BEST_WEIGHTS = f"{results.save_dir}/weights/best.pt"

# %%
# Show training curves
from IPython.display import Image as IPImage, display

results_img = f"{results.save_dir}/results.png"
if os.path.exists(results_img):
    display(IPImage(filename=results_img, width=900))
    print(f"Training curves saved: {results_img}")

# %% [markdown]
# ## Task 03 — Human & Car Detection with Counting

# %% [markdown]
# ### Cell 9 — Detection + Counting on Images

# %%
import cv2
from ultralytics import YOLO as ULT

# Load best trained weights
det_model = ULT(BEST_WEIGHTS)

def detect_and_count(image_path, conf=0.25, iou=0.45, save_path=None):
    """
    Run inference, draw boxes, count humans and cars.
    Returns: annotated image (numpy), human_count, car_count
    """
    results = det_model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        verbose=False
    )[0]

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    human_count = 0
    car_count = 0

    # Colors: human=green, car=red
    COLORS = {0: (34, 197, 94), 1: (239, 68, 68)}
    LABELS = {0: "human", 1: "car"}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == 0:
            human_count += 1
        elif cls_id == 1:
            car_count += 1

        color = COLORS[cls_id]
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Label with confidence
        label = f"{LABELS[cls_id]} {conf_score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - lh - 4), (x1 + lw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Overlay count panel (top-left)
    panel_h, panel_w = 70, 260
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (15, 23, 42), -1)
    alpha = 0.75
    img_region = img[10:10+panel_h, 10:10+panel_w]
    cv2.addWeighted(overlay[10:10+panel_h, 10:10+panel_w], alpha, img_region, 1-alpha, 0, img_region)
    img[10:10+panel_h, 10:10+panel_w] = img_region

    cv2.putText(img, f"Humans : {human_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 197, 94), 2)
    cv2.putText(img, f"Cars   : {car_count}",   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (239, 68, 68), 2)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img, human_count, car_count

# %%
# Run on test samples
test_imgs = sorted(glob.glob(f"{DATASET_DIR}/**/test*/images/*.jpg", recursive=True))
if not test_imgs:
    test_imgs = img_files  # fallback to all images

sample_test = random.sample(test_imgs, min(6, len(test_imgs)))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, img_path in enumerate(sample_test):
    out_path = f"task03_detection_{i+1}.jpg"
    annotated, h_count, c_count = detect_and_count(img_path, save_path=out_path)
    axes[i].imshow(annotated)
    axes[i].set_title(f"Humans: {h_count} | Cars: {c_count}", fontsize=11)
    axes[i].axis("off")
    print(f"Image {i+1}: {h_count} humans, {c_count} cars → saved {out_path}")

plt.suptitle("Task 03 — Human & Car Detection with Counting", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("task03_detection_grid.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: task03_detection_grid.png")

# %% [markdown]
# ### Cell 10 — Process a Full Folder (Batch Inference)

# %%
def batch_inference(folder, output_folder="task03_batch_output", conf=0.25, max_images=50):
    """Run detection + counting on all images in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    img_paths = sorted(glob.glob(f"{folder}/*.jpg"))[:max_images]

    records = []
    for img_path in tqdm(img_paths, desc="Batch inference"):
        fname = os.path.basename(img_path)
        out_path = os.path.join(output_folder, fname)
        _, h, c = detect_and_count(img_path, conf=conf, save_path=out_path)
        records.append({"image": fname, "humans": h, "cars": c})

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(output_folder, "counts.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nBatch complete: {len(records)} images")
    print(f"Results saved to: {csv_path}")
    print(df_results.describe().round(1))
    return df_results

# Run batch on test set (first 50 images)
test_img_folder = os.path.dirname(sample_test[0]) if sample_test else None
if test_img_folder:
    df_batch = batch_inference(test_img_folder, max_images=50)

# %% [markdown]
# ## Task 04 (Bonus) — Object Tracking with ByteTrack

# %% [markdown]
# ### Cell 11 — Track Objects in a Video (or sequence of images)

# %%
# Ultralytics has ByteTrack built-in — no extra install needed
# If you have a drone video, set VIDEO_PATH below.
# If you only have images, we create a short video from the test sequence first.

def images_to_video(img_folder, out_path="sequence.mp4", fps=10):
    """Create a video from a folder of images (sorted by name)."""
    imgs = sorted(glob.glob(f"{img_folder}/*.jpg"))[:100]
    if not imgs:
        print("No images found for video creation.")
        return None
    frame = cv2.imread(imgs[0])
    H, W = frame.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for p in imgs:
        writer.write(cv2.imread(p))
    writer.release()
    print(f"Created video: {out_path} ({len(imgs)} frames @ {fps} fps)")
    return out_path

# Create a test video from image sequence
VIDEO_PATH = "sequence.mp4"
if test_img_folder and not os.path.exists(VIDEO_PATH):
    VIDEO_PATH = images_to_video(test_img_folder)

# %%
if VIDEO_PATH and os.path.exists(VIDEO_PATH):
    track_model = ULT(BEST_WEIGHTS)

    # ByteTrack tracking — built into ultralytics
    track_results = track_model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",   # built-in ByteTrack config
        conf=0.25,
        iou=0.45,
        persist=True,               # maintain track IDs across frames
        save=True,
        project="antlings_runs",
        name="tracking_output",
        exist_ok=True,
        classes=[0, 1],             # human=0, car=1
        verbose=False,
    )
    print("Tracking complete!")
    print(f"Output saved to: antlings_runs/tracking_output/")
else:
    print("No video found for tracking — skipping Task 04.")
    print("Tip: Point VIDEO_PATH to any .mp4 drone footage and rerun this cell.")

# %%
# Custom tracking visualization with track IDs + per-class count
def track_video(video_path, output_path="task04_tracked.mp4", conf=0.25):
    """Track with custom overlay showing track IDs and live counts."""
    model = ULT(BEST_WEIGHTS)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx = 0
    track_history = {}  # track_id → list of center points

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, conf=conf, persist=True, verbose=False)[0]
        human_count = 0
        car_count = 0

        if results.boxes.id is not None:
            for box, track_id, cls_id in zip(
                results.boxes.xyxy,
                results.boxes.id.int(),
                results.boxes.cls.int()
            ):
                x1, y1, x2, y2 = map(int, box)
                tid = int(track_id)
                cid = int(cls_id)

                color = (34, 197, 94) if cid == 0 else (239, 68, 68)
                label_name = "H" if cid == 0 else "C"

                if cid == 0:
                    human_count += 1
                else:
                    car_count += 1

                # Draw box + track ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_name}#{tid}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw track trail
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append((cx, cy))
                if len(track_history[tid]) > 20:
                    track_history[tid].pop(0)
                pts = track_history[tid]
                for j in range(1, len(pts)):
                    alpha = int(255 * j / len(pts))
                    cv2.line(frame, pts[j-1], pts[j], color, 1)

        # Count overlay
        cv2.rectangle(frame, (10, 10), (270, 75), (15, 23, 42), -1)
        cv2.putText(frame, f"Humans : {human_count}", (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 197, 94), 2)
        cv2.putText(frame, f"Cars   : {car_count}", (18, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (239, 68, 68), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (W - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Tracked video saved: {output_path} ({frame_idx} frames)")
    return output_path

if VIDEO_PATH and os.path.exists(VIDEO_PATH):
    track_video(VIDEO_PATH)

# %% [markdown]
# ## Task 05 — Evaluation & Visualization

# %% [markdown]
# ### Cell 12 — Run Validation Metrics (mAP, Precision, Recall)

# %%
eval_model = ULT(BEST_WEIGHTS)

metrics = eval_model.val(
    data=str(yaml_path.resolve()),
    split="val",
    conf=0.25,
    iou=0.5,
    verbose=True,
)

print("\n==============================")
print("EVALUATION RESULTS")
print("==============================")
print(f"mAP@0.5:       {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
print(f"Precision:     {metrics.box.mp:.4f}")
print(f"Recall:        {metrics.box.mr:.4f}")
print(f"\nPer-class mAP@0.5:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:10s}: {metrics.box.ap50[i]:.4f}")

# %%
# Speed benchmark on GPU
import time

speed_model = ULT(BEST_WEIGHTS)
test_imgs_for_speed = sample_test[:20]

start = time.time()
for img_path in test_imgs_for_speed:
    speed_model.predict(img_path, conf=0.25, verbose=False)
elapsed = time.time() - start

fps = len(test_imgs_for_speed) / elapsed
print(f"\nInference Speed:")
print(f"  Images:   {len(test_imgs_for_speed)}")
print(f"  Time:     {elapsed:.2f}s")
print(f"  FPS:      {fps:.1f}")

# %% [markdown]
# ### Cell 13 — Final Visualization Dashboard

# %%
# Grid of predictions with ground truth comparison
def plot_prediction_grid(test_img_paths, n=6):
    """Show model predictions on test images in a grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#0F172A")
    axes = axes.flatten()

    for i, img_path in enumerate(test_img_paths[:n]):
        ann_path = img_path.replace("images", "annotations").replace(".jpg", ".txt")
        annotated, h_count, c_count = detect_and_count(img_path)
        axes[i].imshow(annotated)
        axes[i].set_title(f"Humans: {h_count}  |  Cars: {c_count}",
                          color="white", fontsize=11, pad=6)
        axes[i].axis("off")
        for spine in axes[i].spines.values():
            spine.set_edgecolor("#334155")

    plt.suptitle("Task 05 — Detection & Counting Results",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("task05_final_grid.png", dpi=150, bbox_inches="tight",
                facecolor="#0F172A")
    plt.show()
    print("Saved: task05_final_grid.png")

plot_prediction_grid(sample_test)

# %%
# Metrics summary chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0F172A")

# Per-class mAP bar
classes = CLASS_NAMES
map50_vals = [metrics.box.ap50[i] for i in range(len(CLASS_NAMES))]
colors = ["#22C55E", "#EF4444"]
bars = axes[0].bar(classes, map50_vals, color=colors, edgecolor="#334155", width=0.5)
axes[0].set_ylim(0, 1)
axes[0].set_facecolor("#1E293B")
axes[0].set_title("mAP@0.5 per class", color="white", fontsize=12)
axes[0].tick_params(colors="white")
axes[0].spines[:].set_color("#334155")
for bar, val in zip(bars, map50_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f"{val:.3f}", ha="center", color="white", fontsize=11)

# Overall metrics radar
metrics_dict = {
    "Precision": metrics.box.mp,
    "Recall":    metrics.box.mr,
    "mAP@0.5":  metrics.box.map50,
    "mAP@50:95":metrics.box.map,
}
metric_names = list(metrics_dict.keys())
metric_vals  = list(metrics_dict.values())
bar_colors = ["#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899"]
bars2 = axes[1].barh(metric_names, metric_vals, color=bar_colors, edgecolor="#334155")
axes[1].set_xlim(0, 1)
axes[1].set_facecolor("#1E293B")
axes[1].set_title("Overall Metrics", color="white", fontsize=12)
axes[1].tick_params(colors="white")
axes[1].spines[:].set_color("#334155")
for bar, val in zip(bars2, metric_vals):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color="white", fontsize=11)

plt.tight_layout()
plt.savefig("task05_metrics.png", dpi=150, bbox_inches="tight", facecolor="#0F172A")
plt.show()
print("Saved: task05_metrics.png")

# %% [markdown]
# ### Cell 14 — Strengths, Limitations & Challenges Summary

# %%
print("""
======================================================
TASK 05 — ANALYSIS SUMMARY
======================================================

STRENGTHS
---------
+ YOLOv8 is fast (real-time capable) and accurate for
  mixed-scale object detection.
+ Mosaic + copy-paste augmentation specifically helps
  with VisDrone's small, dense objects.
+ ByteTrack tracking works without re-identification
  models — low overhead, reliable track IDs.
+ Dual-class filtering (human = pedestrian + people)
  gives more complete human counts.

LIMITATIONS
-----------
- Small objects (<10px) are still often missed;
  a higher input resolution (1280) would help.
- Counting is detection-based, not tracking-based —
  so overlapping or fast-moving groups may be
  double-counted or under-counted.
- Model not evaluated on night/foggy conditions.
- Ignored regions (category 0) were excluded but
  their bounding boxes may partially overlap targets.

CHALLENGES FACED
----------------
- VisDrone annotation format differs from YOLO —
  required a custom conversion pipeline.
- Class imbalance: pedestrian class dominates.
  Addressed with mosaic/copy-paste augmentation.
- Dense scenes: IoU-based NMS struggles with
  tightly packed humans; lower NMS threshold helps.
- AMD ROCm + Ultralytics: ensure torch+rocm wheel
  matches your ROCm driver version (check with
  `rocminfo` or `rocm-smi`).

OPTIONAL IMPROVEMENTS
---------------------
* Use SAHI (Slicing Aided Hyper Inference) for
  better small-object detection in high-res images.
* Fine-tune with VisDrone-specific anchor clustering.
* Use track ID persistence for unique person counting.
* Export to ONNX for faster CPU inference on demo.
======================================================
""")

# %% [markdown]
# ### Cell 15 — Export Model + Save All Outputs

# %%
# Collect all saved outputs
import shutil

OUTPUT_DIR = Path("./antlings_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Copy result images
for f in glob.glob("task0*.png") + glob.glob("task0*.jpg"):
    shutil.copy2(f, OUTPUT_DIR / f)

# Copy best weights
shutil.copy2(BEST_WEIGHTS, OUTPUT_DIR / "best.pt")

# Optional: export to ONNX for deployment
try:
    onnx_model = ULT(BEST_WEIGHTS)
    onnx_model.export(format="onnx", imgsz=640, opset=11)
    onnx_path = BEST_WEIGHTS.replace(".pt", ".onnx")
    if os.path.exists(onnx_path):
        shutil.copy2(onnx_path, OUTPUT_DIR / "model.onnx")
        print("ONNX export complete.")
except Exception as e:
    print(f"ONNX export skipped: {e}")

print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")

print("\n============================")
print(" ASSESSMENT COMPLETE ")
print("============================")
print("Push to GitHub:")
print("  antlings_outputs/  → results/")
print("  this notebook      → notebook/antlings_assessment.ipynb")
print("Record 3-5 min demo video showing:")
print("  1. Dataset overview (Task01 plots)")
print("  2. Training curves")
print("  3. Detection + counting on test images")
print("  4. Tracking video (if done)")
print("  5. Metrics summary")