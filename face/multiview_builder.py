
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .multiview_types import (
    MultiViewBin,
    MultiViewConfig,
    MultiViewPersonModel,
    MultiViewSample,
    PoseBin,
    classify_pose_bin,
    posebin_from_hint,
)

logger = logging.getLogger(__name__)


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm > 1e-6:
        v = v / norm
    else:
        v[:] = 0.0
    return v


def _sorted_by_quality(samples: Sequence[MultiViewSample]) -> List[MultiViewSample]:
    return sorted(samples, key=lambda s: float(s.quality), reverse=True)


def _compute_centroid(samples: Sequence[MultiViewSample]) -> Tuple[Optional[np.ndarray], float]:
    if not samples:
        return None, 0.0

    embs = np.stack([_l2_normalise(s.embedding) for s in samples], axis=0)
    centroid = _l2_normalise(np.mean(embs, axis=0))
    avg_q = float(sum(float(s.quality) for s in samples) / len(samples))
    return centroid, avg_q


@dataclass
class MultiViewBuilder:

    cfg: MultiViewConfig


    def build_person_model(
        self,
        person_id: str,
        samples: Iterable[MultiViewSample],
        *,
        notes: Optional[Dict[str, Any]] = None,
    ) -> MultiViewPersonModel:
        self.cfg.validate()
        samples_list = list(samples)

        filtered: List[MultiViewSample] = []
        for s in samples_list:
            if float(s.quality) < self.cfg.min_quality_for_model:
                continue

            if s.pose_bin in (PoseBin.UNKNOWN, PoseBin.OCCLUDED):
                pb = classify_pose_bin(
                    yaw_deg=s.yaw_deg,
                    pitch_deg=s.pitch_deg,
                    cfg=self.cfg,
                    is_occluded=(s.pose_bin == PoseBin.OCCLUDED),
                )
                s = s.clone_with_bin(pb)

            filtered.append(s)

        bins: Dict[PoseBin, List[MultiViewSample]] = {}
        for s in filtered:
            bins.setdefault(s.pose_bin, []).append(s)

        mv_bins: Dict[PoseBin, MultiViewBin] = {}
        for pose_bin, bin_samples in bins.items():
            if not bin_samples:
                continue

            sorted_samples = _sorted_by_quality(bin_samples)
            kept = sorted_samples[: self.cfg.max_samples_per_bin]

            centroid, avg_q = _compute_centroid(kept)
            mv_bins[pose_bin] = MultiViewBin(
                pose_bin=pose_bin,
                samples=kept,
                centroid=centroid,
                avg_quality=avg_q,
            )

        model = MultiViewPersonModel(
            person_id=person_id,
            bins=mv_bins,
            notes=notes or {},
        )
        logger.info(
            "Built MultiViewPersonModel for %s | bins=%d coverage=%.2f",
            person_id,
            model.num_populated_bins(),
            model.coverage_score(),
        )
        return model

    def build_models_for_many(
        self,
        person_to_samples: Mapping[str, Iterable[MultiViewSample]],
    ) -> Dict[str, MultiViewPersonModel]:
        models: Dict[str, MultiViewPersonModel] = {}
        for pid, samples in person_to_samples.items():
            try:
                model = self.build_person_model(pid, samples)
                models[pid] = model
            except Exception as exc:
                logger.exception(
                    "Failed to build MultiViewPersonModel for %s: %s", pid, exc
                )
        return models


    def update_person_model_ema(
        self,
        model: MultiViewPersonModel,
        new_samples: Iterable[MultiViewSample],
    ) -> MultiViewPersonModel:
        self.cfg.validate()
        alpha = float(self.cfg.ema_alpha)

        for s in new_samples:
            if float(s.quality) < self.cfg.min_quality_for_model:
                continue

            pb = classify_pose_bin(
                yaw_deg=s.yaw_deg,
                pitch_deg=s.pitch_deg,
                cfg=self.cfg,
                is_occluded=(s.pose_bin == PoseBin.OCCLUDED),
            )
            s = s.clone_with_bin(pb)

            bin_obj = model.bins.get(pb)
            if bin_obj is None:
                centroid = _l2_normalise(s.embedding)
                model.bins[pb] = MultiViewBin(
                    pose_bin=pb,
                    samples=[s],
                    centroid=centroid,
                    avg_quality=float(s.quality),
                )
                continue

            bin_obj.samples.append(s)
            if len(bin_obj.samples) > self.cfg.max_samples_per_bin:
                bin_obj.samples = _sorted_by_quality(bin_obj.samples)[
                    : self.cfg.max_samples_per_bin
                ]

            new_emb = _l2_normalise(s.embedding)
            if bin_obj.centroid is None:
                bin_obj.centroid = new_emb
            else:
                old = _l2_normalise(bin_obj.centroid)
                mixed = alpha * new_emb + (1.0 - alpha) * old
                bin_obj.centroid = _l2_normalise(mixed)

            _, avg_q = _compute_centroid(bin_obj.samples)
            bin_obj.avg_quality = avg_q

        latest_ts = model.updated_at
        for b in model.bins.values():
            for s in b.samples:
                if s.ts > latest_ts:
                    latest_ts = s.ts
        model.updated_at = float(latest_ts)

        return model


    def samples_from_templates(
        self,
        templates: Sequence[Any],
        *,
        yaw_key: str = "yaw_deg",
        pitch_key: str = "pitch_deg",
        roll_key: str = "roll_deg",
        quality_key: str = "quality",
        default_quality: float = 0.7,
        pose_bin_metadata_key: str = "pose_bin_hint",
        source: str = "gallery_template",
    ) -> List[MultiViewSample]:
        samples: List[MultiViewSample] = []

        for t in templates:
            emb = getattr(t, "embedding", None)
            if emb is None:
                logger.warning("Template without 'embedding' attribute skipped.")
                continue

            meta: Dict[str, Any] = {}
            raw_meta = getattr(t, "metadata", None)
            if isinstance(raw_meta, Mapping):
                meta = dict(raw_meta)

            yaw = meta.get(yaw_key)
            if yaw is None and "yaw" in meta:
                yaw = meta.get("yaw")

            pitch = meta.get(pitch_key)
            if pitch is None and "pitch" in meta:
                pitch = meta.get("pitch")

            roll = meta.get(roll_key)
            if roll is None and "roll" in meta:
                roll = meta.get("roll")

            quality_val = meta.get(quality_key, default_quality)
            if quality_val is None and "face_quality" in meta:
                quality_val = meta.get("face_quality", default_quality)
            quality = float(quality_val)

            pose_bin_hint_val = meta.get(pose_bin_metadata_key)
            if pose_bin_hint_val is None and "pose_bin" in meta:
                pose_bin_hint_val = meta.get("pose_bin")

            pose_bin = posebin_from_hint(pose_bin_hint_val)

            sample = MultiViewSample(
                embedding=np.asarray(emb, dtype=np.float32),
                yaw_deg=float(yaw) if yaw is not None else None,
                pitch_deg=float(pitch) if pitch is not None else None,
                roll_deg=float(roll) if roll is not None else None,
                quality=quality,
                pose_bin=pose_bin,
                source=source,
                metadata=meta,
            )
            samples.append(sample)

        return samples
