
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import torch

from core.device import select_device

logger = logging.getLogger(__name__)

MOTION_ANALYSIS_DIR = Path(__file__).parent.parent.resolve()


@dataclass
class GaitDeviceConfig:
    device: str = "cpu" 
    use_half: bool = False 


@dataclass
class GaitModelConfig:
    pose_model_name: str = "yolov8n-pose_openvino_model/"
    gait_embedding_model_path: str = "models/gait_temporal_encoder.pth"


@dataclass
class GaitThresholdConfig:
    min_visibility: float = 0.4
    min_valid_joints: int = 10
    max_match_distance: float = 0.30
    max_weak_match_distance: float = 0.40
    min_gait_quality: float = 0.5
    min_match_margin: float = 0.10


@dataclass
class GaitRouteConfig:
    process_every_n_frames: int = 3
    max_seconds_lookback: float = 3.0
    max_entries_per_track: int = 60 
    min_sequence_length: int = 30
    keypoint_ema_alpha: float = 0.65
    keypoint_history_length: int = 45 
    img_size: int = 640


@dataclass
class GaitGalleryConfig:
    dim: int = 256
    metric: str = "cosine"
    gallery_path: Path = Path("data/gait_gallery.pkl")
    encryption_key_env: str = "GAITGUARD_GAIT_KEY"
    ema_alpha: float = 0.5


@dataclass
class GaitRobustConfig:
    min_seq_len: int = 30
    eval_period: float = 0.7
    
    quality_min: float = 0.55
    quality_confirm: float = 0.65
    
    threshold_candidate: float = 0.60
    threshold_confirm: float = 0.70
    margin_confirm: float = 0.05
    confirm_streak: int = 2
    
    min_motion: float = 0.05

    anthro_threshold: float = 0.15
    anthro_penalty_weight: float = 0.5


@dataclass
class GaitConfig:
    device: GaitDeviceConfig
    models: GaitModelConfig
    thresholds: GaitThresholdConfig
    route: GaitRouteConfig
    gallery: GaitGalleryConfig
    robust: GaitRobustConfig = field(default_factory=GaitRobustConfig)


def default_gait_config(
    prefer_gpu: bool = True,
    base_dir: Optional[Path] = None,
) -> GaitConfig:
    device_str, use_half = select_device(prefer_gpu=prefer_gpu)
    device_cfg = GaitDeviceConfig(device=device_str, use_half=use_half)

    base_dir = base_dir or MOTION_ANALYSIS_DIR
    
    gait_model_path = (base_dir / "models" / "gait_temporal_encoder.pth").resolve()
    openvino_path = (base_dir / "yolov8n-pose_openvino_model").resolve()
    
    main_data_dir = (base_dir.parent / "data").resolve()
    
    if not gait_model_path.exists():
        logger.warning(f"⚠️ Gait model not found: {gait_model_path}")
    else:
        logger.info(f"✅ Gait model found: {gait_model_path}")
    
    if openvino_path.exists():
        pose_model = str(openvino_path)
        logger.info(f"✅ Using OpenVINO pose model: {openvino_path}")
    else:
        pose_model = "yolov8n-pose.pt"
        logger.warning(f"⚠️ OpenVINO not found at {openvino_path}. Falling back to PyTorch.")

    return GaitConfig(
        device=device_cfg,
        models=GaitModelConfig(
            pose_model_name=pose_model, 
            gait_embedding_model_path=str(gait_model_path)
        ),
        thresholds=GaitThresholdConfig(
            max_match_distance=0.30,
            max_weak_match_distance=0.40,
            min_match_margin=0.10
        ),
        route=GaitRouteConfig(
            min_sequence_length=30, 
            keypoint_history_length=45
        ),
        gallery=GaitGalleryConfig(
            dim=256, 
            gallery_path=(main_data_dir / "gait_gallery.pkl")
        )
    )