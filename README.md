# AnnotateX

**Intelligent Automatic Data Annotation Pipeline for Computer Vision**

AnnotateX is not just "YOLO + UI". It is a quality-aware annotation pipeline that uses YOLO for object detection, then runs every detection through a **deterministic Quality Engine** and **Validation layer** to decide whether that detection is trustworthy enough to become a usable dataset annotation — before exporting clean, ready-to-train datasets in YOLO and COCO formats.

> *"YOLO produces detections, but AnnotateX determines whether those detections are trustworthy enough to become usable annotations — that's the difference between an object detector and an annotation pipeline."*

---

## Problem Statement

Manual data annotation is one of the biggest bottlenecks in training computer vision models. It's slow, expensive, and error-prone. While pre-trained models like YOLO can auto-detect objects, their raw output isn't directly usable as training data — detections can be malformed, duplicated, too small, or low-confidence.

**AnnotateX solves this** by building an intelligent pipeline around YOLO that automatically:
1. **Detects** objects using a pretrained YOLO model
2. **Evaluates** each detection through four deterministic quality rules
3. **Routes** detections as ACCEPT / FLAG / REJECT with human-readable explanations
4. **Validates** accepted annotations for structural/format safety
5. **Exports** clean datasets in industry-standard YOLO `.txt` and COCO `.json` formats

The result: upload images → get a production-ready annotated dataset, with full transparency into why every annotation was kept or dropped.

---

## Tech Stack

| Tool | Purpose | Why This Choice |
|------|---------|-----------------|
| **Python 3.10+** | Core language | Standard for ML/CV projects |
| **Gradio (Customized)** | Web UI | Overhauled into a premium dark-themed SaaS dashboard with custom CSS/JS and Plotly charts, escaping the default Gradio look while keeping Python simplicity |
| **Ultralytics YOLOv8** | Object detection | Pretrained, fast to integrate, gives boxes + class + confidence out of the box |
| **OpenCV** | Image processing | Resize, drawing bounding boxes, array operations |
| **Pillow** | Image validation | Simple format checks and loading |
| **PyYAML** | Configuration | Human-readable threshold tuning without code changes |
| **pytest** | Testing | Standard, minimal setup |
| **Python dataclasses** | Data models | Stdlib, zero dependency, sufficient for a 3-day MVP |

**Deliberately excluded:** FastAPI (no need for a separate API layer — Gradio calls Python directly), Pydantic (dataclasses are simpler for the team), Docker/Kubernetes (overhead for 3 days), any database (in-memory per-run results are enough), any paid API (all models run locally).

---

## Pipeline Workflow

```
  Upload Images
       │
       ▼
  ┌─────────────────┐
  │  PREPROCESSING   │  Validate format, detect corruption, resize if needed
  └────────┬────────┘
           │ clean images
           ▼
  ┌─────────────────┐
  │  YOLO INFERENCE  │  Detect objects → list of Detection(bbox, class, confidence)
  └────────┬────────┘
           │ raw detections
           ▼
  ┌─────────────────┐
  │  QUALITY ENGINE  │  4 rules: confidence, valid bounds, minimum size, no duplicates
  └────────┬────────┘
           │
     ┌─────┼──────┐
     ▼     ▼      ▼
  ACCEPT  FLAG  REJECT      ← every decision is explainable (named rules)
     │     │      │
     │     │      └── dropped, logged
     │     └── shown in UI "needs review" list
     ▼
  ┌─────────────────┐
  │   VALIDATION     │  Structural checks: valid class ID, bbox fits image, no dupes
  └────────┬────────┘
           │ safe annotations only
           ▼
  ┌─────────────────┐
  │     EXPORT       │  YOLO .txt (normalized) + COCO .json (pixel coords)
  └────────┬────────┘
           │
           ▼
     Dashboard + Download
```

---

## Folder Structure

```
AnnotateX/
│
├── app/
│   ├── ui.py                     # Main dashboard layout and page routing
│   ├── components.py             # Reusable HTML/Plotly UI components (sidebar, charts, tables)
│   └── styles.py                 # Comprehensive custom CSS for the dark SaaS theme
│
├── pipeline/
│   └── orchestrator.py           # Runs pipeline stages in order, builds BatchResult
│
├── preprocessing/
│   ├── validate.py               # Image format/corruption checks → ImageInput
│   └── resize.py                 # Resize large images for faster inference
│
├── inference/
│   └── yolo_infer.py             # YOLO model loading + inference → list[Detection]
│
├── quality/
│   ├── rules.py                  # Four quality rules (confidence, bounds, size, duplicates)
│   └── engine.py                 # Combines rules → QualityResult with ACCEPT/FLAG/REJECT
│
├── routing/
│   └── router.py                 # Splits detections into accepted/flagged/rejected lists
│
├── validation/
│   └── validate_export.py        # Pre-export structural/format safety checks
│
├── export/
│   ├── yolo_writer.py            # YOLO .txt export (normalized coordinates)
│   └── coco_writer.py            # COCO .json export (pixel coordinates)
│
├── models/
│   └── contracts.py              # ⚠️ LOCKED shared dataclasses — single source of truth
│
├── utils/
│   ├── iou.py                    # Intersection over Union calculation
│   └── image_utils.py            # Shared image loading/drawing helpers
│
├── configs/
│   └── config.yaml               # Thresholds, class names, paths (tune without code changes)
│
├── sample_data/                  # Test images for development and demo
├── outputs/                      # Exported datasets land here (gitignored)
├── tests/                        # Unit + integration tests (pytest)
│   ├── test_inference.py
│   ├── test_preprocessing.py
│   ├── test_quality.py
│   ├── test_routing.py
│   ├── test_export.py
│   ├── test_validation.py
│   └── test_ui_integration.py
│
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
```

### Key file: `models/contracts.py`

This is the **single source of truth** for all data shapes in the pipeline. Every module imports from it. The shared dataclasses are:

| Dataclass | Purpose |
|-----------|---------|
| `ImageInput` | Represents an uploaded image after validation (status: ok/corrupt/unsupported) |
| `Detection` | A single YOLO detection (image_id, bbox, class_id, class_name, confidence) |
| `QualityResult` | Quality evaluation outcome (passed/failed rules, ACCEPT/FLAG/REJECT decision) |
| `ValidationResult` | Structural validation result (is_valid, list of errors) |
| `ProcessingResult` | Complete outcome for one image (detections + quality + routing results) |
| `BatchResult` | Aggregate result for all images (totals + export paths) |

---

## Quality Engine — The Core Differentiator

Confidence score alone does NOT guarantee a detection is usable. A high-confidence box can still be:
- **Malformed** (outside image bounds, zero area)
- **Duplicated** (same object detected twice)
- **Tiny** (noise, not a real object)

The Quality Engine runs **four deterministic rules** on every detection:

| Rule | What it checks | Fail → |
|------|---------------|--------|
| **Confidence** | `conf >= threshold` (default 0.5) | FLAG |
| **Valid Bounds** | bbox inside image, x2>x1, y2>y1, area>0 | REJECT |
| **Not Tiny** | bbox area >= min_area_px (default 400px) | FLAG |
| **No Duplicate** | IoU with same-class detections < threshold (default 0.9) | FLAG |

Every decision is **explainable** — the QualityResult includes exactly which rules passed and failed, plus a human-readable reason string.

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Install

```bash
# Clone the repo
git clone https://github.com/your-team/AnnotateX.git
cd AnnotateX

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Add Sample Images

Place 5-8 test images in `sample_data/` (see `sample_data/README.md` for recommendations).

### Run the App

```bash
python main.py
```

This launches the Gradio web interface. Open the URL shown in your terminal (usually `http://localhost:7860`).

### Run Tests

```bash
pytest tests/
```

---

## How to Use

The application features a premium multi-page dashboard accessible via the left sidebar:

1. **Dashboard**: View real-time aggregate metrics, processing throughput, detection distribution (donut charts), and confidence histograms.
2. **Upload & Process**: Drag and drop images, run the pipeline, and track progression through the 6-stage engine.
3. **Results**: Inspect side-by-side original and annotated images with color-coded bounding boxes (Green=Accept, Orange=Flag, Red=Reject).
4. **Quality Engine**: Review the active deterministic rules and view a transparent, per-detection audit log explaining exactly why a box was kept or dropped.
5. **Exports**: Download your clean, pre-validated datasets in YOLO (`.txt`) and COCO (`.json`) formats.

---

## Export Formats

### YOLO `.txt` (one file per image)
```
class_id  x_center  y_center  width  height
```
All coordinates normalized to [0, 1].

### COCO `.json` (single file for entire batch)
```json
{
  "info": {"description": "AnnotateX export"},
  "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [x, y, w, h]}],
  "categories": [{"id": 0, "name": "person"}]
}
```
COCO bbox is `[x_min, y_min, width, height]` in **pixel** coordinates (not normalized).

---

## Configuration

Edit `configs/config.yaml` to tune thresholds without changing code:

```yaml
conf_threshold: 0.5      # Minimum confidence to accept
min_area_px: 400          # Minimum bounding box area (pixels)
iou_threshold: 0.9        # IoU threshold for duplicate detection
```

---

## Team

Built in a 3-day hackathon sprint by a 4-person team.

| Role | Owns |
|------|------|
| Member A — Pipeline Lead | Preprocessing + Inference |
| Member B — Quality Engineer | Quality Engine + Routing |
| Member C — Export Engineer | Validation + Export |
| Member D — UI/Integration | Gradio App + Orchestrator |

---

## License

See [LICENSE](LICENSE) for details.
