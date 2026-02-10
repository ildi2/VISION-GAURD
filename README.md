# Vision HSP — Privacy-First Real-Time Identity Surveillance

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## Overview

**Vision HSP** is a streaming computer vision system for real-time person detection, tracking, and identity-conditioned privacy enforcement. The pipeline detects persons via YOLO, tracks them with an OC-SORT association engine, resolves identity through face embedding search against an encrypted gallery, and produces a **delayed, redacted output stream** in which authorized individuals are masked while unknowns remain fully visible.

The system enforces a **fail-closed privacy policy**: once a person is identified as authorized, redaction locks on and cannot be reversed by a single-frame misclassification. A configurable delay buffer (default 3 seconds) ensures that no frame is ever written before the identity engine has had sufficient time to reach a decision.

### Core Capabilities

- **Identity-conditioned redaction** — authorized persons (residents) are silhouette-masked; unknown persons remain visible for security monitoring.
- **GPS-like identity persistence** — face assigns identity, body tracking carries it through occlusion, head turns, and temporary face loss.
- **Fail-closed policy FSM** — four-state machine guarantees redaction cannot be accidentally removed once applied.
- **Instance segmentation + stabilization** — contour-accurate person masks with temporal smoothing, shrink suppression, and TTL reuse to eliminate flicker.
- **Retroactive redaction** — if identity confirmation arrives after a frame was captured, the buffered raw image is re-rendered with correct redaction before writing.
- **JSONL audit trail** — every emitted frame is logged with per-track policy state, redaction method, mask quality, and timing.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  FRAME ACQUISITION                                                    │
│  Camera (USB/RTSP/file) → BGR Frame @ 30 fps                        │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  PERCEPTION LAYER                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────────┐  │
│  │  YOLO11n    │───▶│  OC-SORT    │    │  InsightFace (buffalo_l) │  │
│  │  Detection  │    │  Tracker    │    │  512-d Face Embedding    │  │
│  │  (person)   │    │  (IoU +     │    │  + Quality Gate          │  │
│  └─────────────┘    │  appearance)│    └────────────┬─────────────┘  │
│                     └──────┬──────┘                 │                │
└────────────────────────────┼───────────────────────┼────────────────┘
                             │                       │
                             ▼                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│  IDENTITY ENGINE                                                      │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Gallery Search   │  │   Binding    │  │  Continuity Binder  │   │
│  │  (FAISS cosine)   │  │   FSM (6     │  │  (GPS-like carry,   │   │
│  │  strong / weak    │  │   states)    │  │   appearance EMA,   │   │
│  │  match thresholds │  │              │  │   spatial checks)   │   │
│  └──────────────────┘  └──────────────┘  └──────────────────────┘   │
└──────────────────────────────┬────────────────────────────────────────┘
                               │  IdentityDecision per track
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  PRIVACY PIPELINE                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ Policy FSM │  │ Segmenter  │  │ Stabilizer │  │  Renderer    │  │
│  │ (4 states) │  │ (YOLOv8seg)│  │ (union +   │  │ (silhouette  │  │
│  │ per-track  │  │ per-track  │  │  TTL +     │  │  or blur)    │  │
│  │ SHOW/REDACT│  │ masks      │  │  shrink)   │  │              │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘  │
│        └───────────────┴───────────────┴─────────────────┘          │
│                               │                                      │
│                               ▼                                      │
│                    ┌─────────────────────┐                           │
│                    │   Delay Buffer (3s) │                           │
│                    └──────────┬──────────┘                           │
│                               │                                      │
│              ┌────────────────┼────────────────┐                    │
│              ▼                ▼                ▼                     │
│       ┌──────────┐    ┌──────────┐    ┌──────────────┐             │
│       │  Writer  │    │  Audit   │    │   Metrics    │             │
│       │ (MP4)    │    │ (JSONL)  │    │  (M6 eval)   │             │
│       └──────────┘    └──────────┘    └──────────────┘             │
└───────────────────────────────────────────────────────────────────────┘
```

**Two-window output:**

| Window | Content | Persisted? |
|--------|---------|------------|
| Raw Preview | Live feed with bounding boxes, identity labels, binding state, FPS | **Never saved** |
| Privacy Output (Delayed) | Redacted stream with silhouettes on authorized persons, unknowns visible | Written to MP4 + JSONL audit |

---

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (recommended for GPU acceleration)
- 6 GB+ VRAM (RTX 3050 or better)

### Setup

```bash
# Clone
git clone https://github.com/ildi2/VISION-GAURD.git
cd VISION-GAURD

# Virtual environment
python -m venv .venv310
.venv310\Scripts\activate          # Windows
# source .venv310/bin/activate     # Linux / macOS

# Dependencies
pip install -r requirements-gpu.txt    # GPU (CUDA)
pip install -r requirements-cpu.txt    # CPU only
```

### Model Weights

Place the following in the project root (excluded from Git via `.gitignore`):

| Model | Purpose | Source |
|-------|---------|--------|
| `yolo11n.pt` | Person detection | [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) |
| `yolov8n-seg.pt` | Instance segmentation (privacy masks) | [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) |
| `yolov8n-pose.pt` | Pose estimation (motion analysis) | [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) |

InsightFace `buffalo_l` is downloaded automatically on first run.

---

## Usage

### Run

```bash
python -m core.main_loop
```

### Configuration

All parameters are in `config/default.yaml`:

```yaml
camera:
  index: 0
  width: 640
  height: 480
  fps: 30

privacy:
  enabled: true
  redaction_style: "silhouette"     # "silhouette" or "blur"
  delay_sec: 3.0                    # seconds held before writing

  policy:
    grace_sec: 5.0                  # redaction grace on signal loss
    reacquire_sec: 10.0             # redaction hold on disappearance
    unlock_allowed: false           # fail-closed (cannot un-redact)
    authorized_categories:
      - "resident"

  segmentation:
    enabled: true
    backend: "yolo_seg"

  stabilization:
    enabled: true
    method: "union_only"
    mask_ttl_sec: 0.75
    max_shrink_ratio: 0.85
```

### Face Enrollment

```bash
python -m identity.enrollment_cli
```

Enrolled templates are encrypted (AES-GCM) and stored in `data/face_gallery.enc`. No raw face images are saved.

---

## Project Structure

```
Vision_HSP/
├── core/                        # Pipeline orchestration
│   ├── main_loop.py             # Main entry point
│   ├── config.py                # Configuration dataclasses
│   ├── camera.py                # Frame acquisition
│   ├── interfaces.py            # Abstract engine contracts
│   ├── scheduler.py             # GPU-budget face scheduler
│   ├── metrics.py               # Runtime face metrics
│   └── governance_metrics.py    # System-wide metric collection
│
├── perception/                  # Detection and tracking
│   ├── detector.py              # YOLO11n person detector
│   ├── tracker_ocsort.py        # OC-SORT tracker (IoU + appearance)
│   ├── appearance.py            # Grid-color histogram features
│   ├── perception_engine.py     # Perception pipeline wrapper
│   └── ring_buffer.py           # Per-track temporal buffer
│
├── face/                        # Face processing
│   ├── detector_align.py        # InsightFace detection + alignment
│   ├── embedder.py              # 512-d embedding normalization
│   ├── quality.py               # Quality scoring (blur, pose, light)
│   ├── route.py                 # Per-track face processing route
│   ├── config.py                # Face thresholds and parameters
│   └── multiview_*.py           # Multi-view pose-bin gallery
│
├── identity/                    # Identity resolution
│   ├── identity_engine.py       # Face-based identity engine
│   ├── face_gallery.py          # Encrypted FAISS gallery
│   ├── binding.py               # 6-state binding FSM
│   ├── continuity_binder.py     # GPS-like identity carry
│   ├── evidence_gate.py         # Quality gate for face evidence
│   ├── merge_manager.py         # Track-fragment merge logic
│   └── crypto.py                # AES-GCM gallery encryption
│
├── privacy/                     # Privacy enforcement pipeline
│   ├── pipeline.py              # Main privacy orchestrator
│   ├── policy_fsm.py            # 4-state redaction policy FSM
│   ├── segmenter.py             # Instance segmentation backend
│   ├── mask_stabilizer.py       # Temporal mask stabilization
│   ├── delay_buffer.py          # Frame delay buffer
│   ├── writer.py                # MP4 video writer
│   ├── audit.py                 # JSONL audit logger
│   └── metrics.py               # Leakage / flicker / timing eval
│
├── schemas/                     # Data structures
│   ├── frame.py                 # Frame container
│   ├── tracklet.py              # Track state
│   ├── identity_decision.py     # Identity output
│   ├── id_signals.py            # Per-track evidence signals
│   └── face_sample.py           # Single face observation
│
├── vision_identity/             # Multi-modal fusion engine
│   ├── simple_fusion.py         # Face + gait fusion logic
│   ├── identity_matcher.py      # Cross-modal matching
│   └── adapters/                # Per-modality adapters
│
├── motion_analysis/             # Gait and pose analysis
│   ├── gait/                    # Gait embedding extraction
│   ├── perception_gait/         # Pose-specific detection
│   └── schemas_gait/            # Gait data structures
│
├── source_auth/                 # Source authenticity verification
│   ├── engine.py                # Screen / spoof detection
│   ├── background.py            # Background consistency
│   └── motion.py                # Motion analysis for liveness
│
├── ui/                          # Display
│   └── overlay.py               # Bounding box + identity overlay
│
├── config/
│   └── default.yaml             # Full system configuration
│
├── scripts/                     # Utilities and rollout stages
├── tests/                       # Unit and integration tests
└── data/                        # Runtime data (gitignored)
```

---

## Key Design Decisions

### Identity Pipeline

The identity engine resolves each tracked person through a multi-stage process:

1. **Detection** — YOLO11n detects all persons in the frame.
2. **Tracking** — OC-SORT assigns persistent `track_id` using IoU + appearance similarity.
3. **Face route** — for each track, a head-region crop is passed to InsightFace for embedding extraction (rate-limited to 1-in-5 frames).
4. **Gallery search** — cosine distance against enrolled templates. Strong match ≤ 0.85, weak match ≤ 0.93.
5. **Binding FSM** — noisy per-frame matches are smoothed into stable identity assignments over 6 states (UNKNOWN → PENDING → CONFIRMED_WEAK → CONFIRMED_STRONG → STALE → SWITCH_PENDING).
6. **Continuity carry** — when the face is not visible, the binder carries the last confirmed identity using spatial proximity and appearance EMA checks.

### Privacy Enforcement

The privacy pipeline applies **identity-conditioned redaction** with a 4-state per-track policy FSM:

| State | Action | Meaning |
|-------|--------|---------|
| `UNKNOWN_VISIBLE` | SHOW | Not yet identified — visible for security. |
| `AUTHORIZED_LOCKED_REDACT` | REDACT | Confirmed authorized — silhouette applied. |
| `REACQUIRE_REDACT` | REDACT | Track lost while locked — stays redacted. |
| `ENDED_COOLDOWN` | REDACT | Reacquire expired — re-locks if seen again. |

**Fail-closed invariant:** `unlock_allowed = false` by default. Once redaction locks, no classification error can remove it.

### Mask Pipeline

1. **Segmentation** — YOLOv8n-seg produces per-person binary masks, associated to tracks via IoU.
2. **Stabilization** — temporal union of mask history prevents flicker; shrink suppression rejects area drops > 15%; TTL cache reuses masks for up to 0.75 s during occlusion.
3. **Rendering** — silhouette (black fill) or Gaussian blur applied within mask contour. Bounding-box blur fallback if segmentation fails.
4. **Delay buffer** — frames are held for 3 seconds, then re-rendered with current FSM state for retroactive correctness.

---

## Metrics and Evaluation

The system continuously evaluates its own privacy effectiveness (`privacy/metrics.py`):

| Metric | Definition |
|--------|------------|
| **Residual detectability** | Faces detected (Haar cascade) inside redacted regions on the emitted frame. Must be zero. |
| **Escape leakage** | An unknown person incorrectly masked. Structurally prevented by policy FSM. |
| **Flicker** | Frame-to-frame mask IoU (mean and p05). Stabilization targets IoU > 0.85. |
| **Time-to-lock** | Seconds from first track appearance to FSM lock. |
| **Time-to-redacted-emit** | Seconds from first appearance to first redacted frame written (includes buffer delay). |
| **Utility** | Ratio of visible (useful) tracks to total tracks per frame. |

All metrics are written to `privacy_output/privacy_metrics.jsonl`.

---

## Privacy and Security

- Face embeddings are **encrypted at rest** (AES-GCM, key from environment variable).
- **No raw face images** are stored — only normalized 512-d vectors.
- The **raw camera preview is never saved** to disk.
- Gallery files (`.enc`, `.pkl`) and model weights (`.pt`) are excluded from version control.
- Every emitted frame produces a **JSONL audit record** with full per-track redaction provenance.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO detection and segmentation models
- [InsightFace](https://github.com/deepinsight/insightface) — Face recognition (buffalo_l)
- [OC-SORT](https://github.com/noahcao/OC_SORT) — Observation-centric tracking
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- [OpenCV](https://opencv.org/) — Image processing and video I/O

---

<p align="center"><b>Vision HSP — Privacy by Design, Identity by Evidence</b></p>
