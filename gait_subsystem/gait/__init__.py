# gait_subsystem/gait/__init__.py
"""
Gait recognition module.

Components:
- GaitExtractor: Pose sequence → 256-dim embedding
- GaitGallery: FAISS-based identity database
- GaitEngine: Orchestrates extraction and matching
"""

from gait_subsystem.gait.config import GaitConfig, default_gait_config
from gait_subsystem.gait.gait_engine import GaitEngine

__all__ = ["GaitConfig", "default_gait_config", "GaitEngine"]