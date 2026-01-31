# Vision_HSP - GaitGuard Biometric Surveillance System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-11.8+-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## Overview

**GaitGuard** is a real-time multi-modal biometric surveillance system that combines **face recognition** with **GPS-like identity persistence** through body tracking. The system maintains continuous person identification even when faces are temporarily occluded, turned away, or obscured.

### Key Innovation: Chimeric Continuity Mode

Unlike traditional systems that lose identity when a face is not visible, GaitGuard implements a **GPS-like carry mechanism**:

- **[F] Face Mode**: Identity assigned via face recognition (InsightFace buffalo_l)
- **[G] GPS Carry Mode**: Identity maintained via body tracking (OC-SORT + YOLO11n)

This enables continuous identification through:
- Head turns and profile views
- Temporary face occlusions (hands, objects)
- Walking away from camera
- Poor lighting conditions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRAME ACQUISITION                           │
│                    Camera/Video → RGB Frame                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PERCEPTION LAYER                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │   YOLO11n       │    │   OC-SORT       │    │  InsightFace   │  │
│  │   Detection     │───▶│   Tracking      │    │  Recognition   │  │
│  │   (persons)     │    │   (track_ids)   │    │  (embeddings)  │  │
│  └─────────────────┘    └────────┬────────┘    └───────┬────────┘  │
└──────────────────────────────────┼─────────────────────┼────────────┘
                                   │                     │
                                   ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    IDENTITY ENGINE (MultiView)                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  5-Pose Bin Gallery: FRONT, LEFT, RIGHT, UP, DOWN           │   │
│  │  Per-bin embeddings with confidence scoring                  │   │
│  │  Strength levels: STRONG (>0.6), WEAK (0.45-0.6), NONE      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTINUITY BINDER (GPS Policy)                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  5 Guards: Confidence, Cooldown, Conflict, Spoof, Memory    │   │
│  │  States: CONFIRMED_STRONG, CONFIRMED_WEAK, GPS_CARRY, etc.  │   │
│  │  Decisions: BIND (face visible) or CARRY (face not visible) │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         UI OVERLAY                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  GREEN box + [F]: Face confirmed identity                   │   │
│  │  CYAN box  + [G]: GPS carry (tracking maintains identity)   │   │
│  │  ORANGE box: Pending identification                         │   │
│  │  GRAY box: Unknown/unidentified                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Features

### Face Recognition
- **InsightFace buffalo_l** model (512-dim embeddings)
- Multi-view gallery with 5 pose bins
- Quality-aware enrollment with threshold filtering
- Anti-spoofing integration (Silent-Face)

### Body Tracking
- **YOLO11n** for person detection
- **OC-SORT** for robust multi-object tracking
- Appearance features for re-identification
- IoU threshold: 0.2, Appearance lambda: 0.4

### GPS-Like Identity Persistence
- Seamless BIND ↔ CARRY transitions
- Configurable memory age limits
- Confidence decay during carry
- Grace period for face reattachment

### Real-Time Performance
- ~5-6 FPS on RTX 3050 (6GB VRAM)
- Minimal GPU memory footprint (~0.05GB)
- Efficient scheduler for face processing

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- RTX 3050 or better recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/marildoC/Vision_HSP.git
cd Vision_HSP

# Create virtual environment
python -m venv .venv310
.venv310\Scripts\activate  # Windows
# source .venv310/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements-gpu.txt  # For GPU
# pip install -r requirements-cpu.txt  # For CPU only

# Download YOLO models (place in root directory)
# - yolo11n.pt
# - yolov8n-pose.pt
```

### Model Downloads
Models are not included in the repository due to size. Download from:
- **YOLO11n**: [Ultralytics](https://docs.ultralytics.com/models/yolo11/)
- **InsightFace buffalo_l**: Auto-downloaded on first run

---

## 🎮 Usage

### Run the Main System

```bash
python -m core.main_loop
```

### Configuration

Edit `config/default.yaml`:

```yaml
# Camera settings
camera:
  source: 0  # 0 for webcam, or path to video file
  width: 1280
  height: 720

# Identity settings
identity:
  chimeric_mode: "continuity"  # Enable GPS-like persistence
  multiview: true
  
# Thresholds
thresholds:
  strong_match: 0.60
  weak_match: 0.45
  quality_min: 0.55
```

### Keyboard Controls
- `Q` or `ESC`: Quit
- `E`: Enroll new person
- `R`: Reset galleries

---

## 📁 Project Structure

```
Vision_HSP/
├── core/                    # Core system components
│   ├── main_loop.py         # Main pipeline orchestrator
│   ├── camera.py            # Camera/video acquisition
│   ├── scheduler.py         # Face processing scheduler
│   ├── metrics.py           # Performance metrics
│   └── governance_metrics.py # System governance
│
├── identity/                # Identity management
│   ├── continuity_binder.py # GPS-like persistence (KEY FILE)
│   ├── identity_engine_multiview.py # Multi-view face matching
│   └── track_state.py       # Per-track state management
│
├── face/                    # Face processing
│   ├── detector_align.py    # Face detection & alignment
│   ├── embedder.py          # Face embedding extraction
│   ├── quality.py           # Face quality assessment
│   └── multiview_*.py       # Multi-view gallery system
│
├── perception/              # Detection & tracking
│   └── tracker.py           # OC-SORT wrapper
│
├── ui/                      # User interface
│   └── overlay.py           # Bounding box & info overlay
│
├── chimeric_identity/       # Chimeric fusion system
│   └── identity_matcher.py  # Face-gait fusion
│
├── config/                  # Configuration
│   └── default.yaml         # Default settings
│
├── docs/                    # Documentation
│   └── PHASE9_RUNBOOK.md    # Deployment guide
│
└── data/                    # Data storage (gitignored)
    ├── face_gallery.enc     # Encrypted face embeddings
    └── identity_registry.json # Person metadata
```

---

## 🔧 Key Components

### ContinuityBinder (`identity/continuity_binder.py`)

The heart of the GPS-like system. Implements:

```python
# Decision flow
if face_visible_and_quality_ok:
    decision = self._bind(...)      # Face-based identity
    decision.id_source = "F"
else:
    decision = self._carry(...)     # GPS-like carry
    decision.id_source = "G"
    decision.binding_state = "GPS_CARRY"
```

### 5 Guards System

| Guard | Purpose |
|-------|---------|
| **Confidence Guard** | Minimum score threshold (0.45) |
| **Cooldown Guard** | Prevents rapid ID switching (30 frames) |
| **Conflict Guard** | Detects multi-person ID collisions |
| **Spoof Guard** | Anti-spoofing validation |
| **Memory Guard** | Limits carry duration (300 frames) |

---

## 📊 Performance Metrics

From live testing:
```
FPS=5.9 | tracks=1 | alerts=0 | GPU: 0.05GB/6.44GB (0.8%)
Governance: faces=1 | binding: {'GPS_CARRY': 1} | scheduler: 1/1
```

- **Face Detection**: ~200ms per frame (scheduled)
- **Tracking**: Real-time (~30ms)
- **Identity Lookup**: ~5ms

---

## 🛡️ Privacy & Security

- Face embeddings are **encrypted** (AES-256)
- No raw face images stored
- Gallery data is **excluded from Git**
- All biometric data stays local

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [InsightFace](https://github.com/deepinsight/insightface)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [Silent-Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

---

<p align="center">
  <b>GaitGuard - Continuous Identity, Uninterrupted Security</b>
</p>
