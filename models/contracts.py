# models/contracts.py — canonical shared data model. Do not redefine these
# elsewhere. Do not change field names/types without a team sync.

from dataclasses import dataclass, field


@dataclass
class ImageInput:
    """Represents a single uploaded image after initial validation.
    
    Created by the Preprocessing layer. Status indicates whether the
    image is usable ("ok") or why it was rejected ("corrupt" / "unsupported_format").
    """
    image_id: str
    file_path: str
    width: int = 0
    height: int = 0
    status: str = "ok"  # "ok" | "corrupt" | "unsupported_format"


@dataclass
class Detection:
    """A single object detection produced by YOLO inference.
    
    Created by the Inference layer, read-only afterward.
    bbox is [x1, y1, x2, y2] in absolute pixel coordinates.
    """
    image_id: str
    bbox: list        # [x1, y1, x2, y2] pixel coords
    class_id: int
    class_name: str
    conf: float


@dataclass
class QualityResult:
    """Result of evaluating a single Detection through the Quality Engine.
    
    Records which rules passed/failed and the routing decision
    (ACCEPT / FLAG / REJECT) with a human-readable reason.
    """
    detection: Detection
    passed_rules: list
    failed_rules: list
    decision: str      # "ACCEPT" | "FLAG" | "REJECT"
    reason: str


@dataclass
class ValidationResult:
    """Result of structural/export-safety validation on a single annotation.
    
    Checked before export — ensures the annotation is safe to write to
    a YOLO .txt or COCO .json file. errors list is empty when is_valid=True.
    """
    annotation: Detection
    is_valid: bool
    errors: list = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Complete processing outcome for a single image.
    
    Aggregates raw detections, quality decisions, and the routed
    accepted/flagged/rejected lists for one image.
    """
    image_id: str
    detections: list
    quality_results: list
    accepted: list
    flagged: list
    rejected: list
    processing_time_ms: float


@dataclass
class BatchResult:
    """Aggregate result for an entire batch of images.
    
    Contains per-image ProcessingResults plus batch-level totals
    and paths to exported dataset files.
    """
    results: list
    total_images: int
    total_detections: int
    total_accepted: int
    total_flagged: int
    total_rejected: int
    export_paths: dict = field(default_factory=dict)
