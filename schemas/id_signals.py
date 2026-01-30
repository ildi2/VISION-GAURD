# schemas/id_signals.py
#
# Canonical identity-related signals that flow into the identity engine.
#
# IMPORTANT for 3D:
#   - FaceSample is the single, rich description of one face observation.
#   - IdSignals.best_face is the primary face sample used for identity.
#   - IdSignals.recent_faces is an optional history for this track.
#
# Legacy fields (face_embedding, face_quality, raw_face_box, pose_bin_hint)
# are kept for backwards compatibility and are derived from best_face
# when possible.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# FaceSample lives in its own module and is re-exported from schemas.__init__.
# We import it here instead of redefining to avoid duplicate class definitions.
from .face_sample import FaceSample


@dataclass
class IdSignals:
    """
    All identity-related features for a given track at a given moment.

    This structure is what the IdentityEngine receives from FaceRoute
    (and later, gait + appearance routes). It can be extended safely as
    long as existing fields are not removed.

    Primary modern view (for 2D + 3D):

      - track_id       : int
      - best_face      : FaceSample or None
      - recent_faces   : optional small history of FaceSample
      - gait_embedding : temporal gait features
      - gait_quality   : 0–1
      - appearance_embedding : clothing / color features
      - appearance_quality   : 0–1

    Legacy/compatibility fields (still supported):

      - face_embedding : kept for older code; mirrors best_face.embedding
      - face_quality   : kept; mirrors best_face.quality
      - extra          : free-form dict, kept but best_face.extra is preferred
      - raw_face_box   : kept; mirrors best_face.bbox
      - pose_bin_hint  : kept; mirrors best_face.pose_bin

    Multiview / 3D extensions should prefer to read:

      - best_face.yaw / pitch / roll
      - best_face.pose_bin
      - best_face.quality
      - best_face.det_score (if stored in FaceSample.extra)
    """

    track_id: int

    # -------------------- FACE (modern canonical) --------------------
    # Best (most informative) face sample for this track at this decision.
    best_face: Optional[FaceSample] = None

    # Optional history of recent face samples for temporal logic / learning.
    recent_faces: List[FaceSample] = field(default_factory=list)

    # -------------------- FACE (legacy fields) -----------------------
    # These are kept for backwards compatibility with older parts of the
    # codebase. New code should prefer best_face/FaceSample but is free
    # to keep these in sync for simplicity.
    face_embedding: Optional[np.ndarray] = None
    face_quality: float = 0.0

    # -------------------- GAIT --------------------
    gait_embedding: Optional[np.ndarray] = None
    gait_quality: float = 0.0

    # -------------------- APPEARANCE --------------
    appearance_embedding: Optional[np.ndarray] = None
    appearance_quality: float = 0.0

    # -------------------- MULTIVIEW EXTRAS --------
    # Generic dict for any extra face/gait/appearance signals that do
    # not yet justify a top-level field.
    extra: Optional[Dict[str, Any]] = None

    # Raw face box used during enrollment/matching diagnostics.
    # Should be consistent with best_face.bbox when available.
    raw_face_box: Optional[Tuple[float, float, float, float]] = None

    # Early pose bin suggestion from the face route (front, left, etc.)
    # Should mirror best_face.pose_bin when available.
    pose_bin_hint: Optional[str] = None

    # ================================================================
    # SOURCE AUTHENTICITY (SourceAuth) FIELDS
    # ================================================================
    # These fields enable SourceAuth to determine if the face is from a
    # real 3D head or a spoof (phone/screen/printed photo).
    #
    # DATA FLOW:
    #   FaceDetectorAligner → FaceEvidence → IdSignals → SourceAuthEngine
    #          landmarks_2d ✅        ✅          ✅ (HERE)     reads ✅
    #
    # CONNECTS TO:
    #   - source_auth/engine.py: _maybe_update_landmark_history() reads these
    #   - source_auth/motion.py: compute_3d_motion_score() analyzes landmarks
    #   - source_auth/fusion.py: fuse_source_auth() uses motion reliability
    #
    # Without these fields, SourceAuth remains UNCERTAIN forever because:
    #   has_motion = False (no landmarks to analyze)
    #   has_background = False (no frame history)
    #   → fusion returns state="UNCERTAIN" (neutral score 0.5)
    # ================================================================
    
    # 2D face landmarks for SourceAuth motion analysis.
    # Expected shape: (5, 2) or (N, 2) numpy array of float32.
    # The 5 landmarks are typically: left_eye, right_eye, nose, left_mouth, right_mouth
    # Used by motion.py to compute parallax ratio (3D vs planar motion).
    landmarks_2d: Optional[np.ndarray] = None
    
    # Face bounding box in full-frame coordinates for SourceAuth.
    # Format: (x1, y1, x2, y2) in pixels.
    # Used to estimate face size for adaptive motion thresholds.
    face_bbox_in_frame: Optional[Tuple[float, float, float, float]] = None

    # ------------------------------------------------------------------
    # Helper methods to keep canonical + legacy views in sync
    # ------------------------------------------------------------------

    @property
    def has_face(self) -> bool:
        """
        Quick check: do we have any face evidence (canonical or legacy)?
        """
        if self.best_face is not None and self.best_face.embedding is not None:
            return True
        if self.face_embedding is not None:
            return True
        return False

    def sync_from_best_face(self) -> None:
        """
        Mirror best_face into legacy fields.

        Call this after best_face is set/updated so that older parts
        of the system that still read legacy fields stay consistent.
        """
        bf = self.best_face
        if bf is None:
            # If canonical face is cleared, also clear legacy mirrors.
            self.face_embedding = None
            self.face_quality = 0.0
            self.raw_face_box = None
            self.pose_bin_hint = None
            return

        # Normalise embedding to 1-D float32 if present.
        if bf.embedding is not None:
            self.face_embedding = np.asarray(bf.embedding, dtype=np.float32).reshape(-1)
        else:
            self.face_embedding = None

        # Quality is always stored as float in [0,1] (clamped for safety).
        q = float(bf.quality)
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        self.face_quality = q

        # Box + pose bin mirrors.
        self.raw_face_box = bf.bbox
        self.pose_bin_hint = bf.pose_bin

        # Merge extra diagnostics: prefer canonical FaceSample.extra,
        # but don't throw away any pre-existing IdSignals.extra keys.
        if bf.extra:
            if self.extra:
                merged = dict(self.extra)
                merged.update(bf.extra)
                self.extra = merged
            else:
                self.extra = dict(bf.extra)

    def set_best_face(self, face: FaceSample) -> None:
        """
        Convenience setter: assign best_face and update legacy mirrors.
        """
        self.best_face = face
        self.sync_from_best_face()

    def add_recent_face(self, face: FaceSample, max_len: int = 5) -> None:
        """
        Append a face sample to recent_faces with a fixed maximum length.

        This is useful for temporal fusion logic (e.g. multiview + gait).
        """
        self.recent_faces.append(face)
        if len(self.recent_faces) > max_len:
            self.recent_faces.pop(0)

    # ------------------------------------------------------------------
    # Optional: backfill best_face from legacy fields (for old routes)
    # ------------------------------------------------------------------

    def ensure_best_face_from_legacy(self, ts: Optional[float] = None) -> None:
        """
        If best_face is missing but legacy face_embedding / raw_face_box /
        extra / pose_bin_hint exist, create a minimal FaceSample so that
        3D / multiview code can operate on canonical structure.

        This is *non-breaking* and is only used when best_face is None.
        """
        if self.best_face is not None:
            return

        if self.face_embedding is None:
            return

        # Build a minimal canonical sample from legacy fields.
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
        # Keep all mirrors consistent.
        self.sync_from_best_face()


# =============================================================================
# SINGLE-MODALITY IDENTITY SIGNAL (for gait/appearance routes)
# =============================================================================

@dataclass
class IdSignal:
    """
    Represents a single identity signal from one recognition modality.

    This is used by the gait engine and future appearance routes to produce
    per-track identity suggestions that are later fused with face evidence.

    Attributes
    ----------
    track_id : int
        The ID of the track this signal pertains to.
    identity_id : Optional[str]
        The suggested person ID. None if unknown/below threshold.
    confidence : float
        Confidence score (0-1) for this identity suggestion.
    method : str
        The modality that produced this signal: "face", "gait", "appearance"
    """
    track_id: int
    identity_id: Optional[str]
    confidence: float
    method: str
    extra: Optional[Dict[str, Any]] = None
