
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from .face_sample import FaceSample


@dataclass
class IdSignals:

    track_id: int

    best_face: Optional[FaceSample] = None

    recent_faces: List[FaceSample] = field(default_factory=list)

    face_embedding: Optional[np.ndarray] = None
    face_quality: float = 0.0

    gait_embedding: Optional[np.ndarray] = None
    gait_quality: float = 0.0

    appearance_embedding: Optional[np.ndarray] = None
    appearance_quality: float = 0.0

    extra: Optional[Dict[str, Any]] = None

    raw_face_box: Optional[Tuple[float, float, float, float]] = None

    pose_bin_hint: Optional[str] = None

    
    landmarks_2d: Optional[np.ndarray] = None
    
    face_bbox_in_frame: Optional[Tuple[float, float, float, float]] = None


    @property
    def has_face(self) -> bool:
        if self.best_face is not None and self.best_face.embedding is not None:
            return True
        if self.face_embedding is not None:
            return True
        return False

    def sync_from_best_face(self) -> None:
        bf = self.best_face
        if bf is None:
            self.face_embedding = None
            self.face_quality = 0.0
            self.raw_face_box = None
            self.pose_bin_hint = None
            return

        if bf.embedding is not None:
            self.face_embedding = np.asarray(bf.embedding, dtype=np.float32).reshape(-1)
        else:
            self.face_embedding = None

        q = float(bf.quality)
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        self.face_quality = q

        self.raw_face_box = bf.bbox
        self.pose_bin_hint = bf.pose_bin

        if bf.extra:
            if self.extra:
                merged = dict(self.extra)
                merged.update(bf.extra)
                self.extra = merged
            else:
                self.extra = dict(bf.extra)

    def set_best_face(self, face: FaceSample) -> None:
        self.best_face = face
        self.sync_from_best_face()

    def add_recent_face(self, face: FaceSample, max_len: int = 5) -> None:
        self.recent_faces.append(face)
        if len(self.recent_faces) > max_len:
            self.recent_faces.pop(0)


    def ensure_best_face_from_legacy(self, ts: Optional[float] = None) -> None:
        if self.best_face is not None:
            return

        if self.face_embedding is None:
            return

        emb = np.asarray(self.face_embedding, dtype=np.float32).reshape(-1)
        extra = self.extra or {}

        self.best_face = FaceSample(
            bbox=self.raw_face_box,
            embedding=emb,
            det_score=float(extra.get("det_score", 0.0)),
            quality=float(self.face_quality),
            yaw=float(extra.get("yaw", 0.0)) if "yaw" in extra else None,
            pitch=float(extra.get("pitch", 0.0)) if "pitch" in extra else None,
            roll=float(extra.get("roll", 0.0)) if "roll" in extra else None,
            ts=ts,
            pose_bin=self.pose_bin_hint,
            source="legacy",
            extra=extra if extra else None,
        )
        self.sync_from_best_face()


@dataclass
class IdSignal:
    track_id: int
    identity_id: Optional[str]
    confidence: float
    method: str
    extra: Optional[Dict[str, Any]] = None
