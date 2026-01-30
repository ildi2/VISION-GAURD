# face/multiview_gallery_view.py
#
# Read-only pseudo-3D / multi-view view over the encrypted FaceGallery.
#
# Responsibilities (Layer-1 multi-view):
#   - Read all PersonEntry + FaceTemplate objects from identity.face_gallery.
#   - Convert each template into MultiViewSample objects using pose/quality
#     metadata that enrollment & face route already attach.
#   - Build a MultiViewPersonModel per person via MultiViewBuilder
#     (pose bins + centroids + quality stats).
#   - Keep everything in memory only; the underlying gallery file format
#     is untouched. If this layer fails, the classic flat-template pipeline
#     still works.
#
# This module does NOT:
#   - Modify or save the FaceGallery.
#   - Do temporal identity smoothing.
#   - Decide which person a live sample belongs to.
#
# Higher layers (identity/multiview_matcher.py and
# identity/identity_engine_multiview.py) will use this as a read-only
# “3D head database”.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from face.multiview_types import (
    MultiViewConfig,
    MultiViewPersonModel,
)
from face.multiview_builder import MultiViewBuilder

# We only need FaceGallery for reading persons/templates.
# Import is runtime, but we keep type hints behind TYPE_CHECKING
# to avoid circular import issues in static tooling.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from identity.face_gallery import FaceGallery, PersonEntry, FaceTemplate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PersonMultiViewSummary:
    """
    Lightweight summary of a person's multi-view coverage.

    Useful for diagnostics, CLI tools, and debugging.
    """

    person_id: str
    num_bins_filled: int
    total_samples: int
    avg_quality: float
    present_bins: List[str]


# ---------------------------------------------------------------------------
# MultiViewGalleryView
# ---------------------------------------------------------------------------


class MultiViewGalleryView:
    """
    Read-only, in-memory pseudo-3D view over a FaceGallery.

    Core ideas:

      * The *gallery* remains the single source of truth: it stores
        encrypted templates + metadata on disk.

      * This class *derives* MultiViewPersonModel objects from those
        templates via MultiViewBuilder.

      * Models are rebuilt lazily when marked dirty, so callers can
        explicitly say "gallery has changed, please refresh" without
        coupling this class to internal gallery implementation details.

    Typical usage:

        gallery = FaceGallery(...)
        mv_view = MultiViewGalleryView(gallery)

        # After enrollment / reset:
        mv_view.mark_dirty()

        # When identity engine starts:
        mv_view.refresh_if_needed()

        model = mv_view.get_person_model("p_0001")
        all_models = mv_view.get_all_models()
    """

    def __init__(
        self,
        gallery: "FaceGallery",
        multiview_cfg: Optional[MultiViewConfig] = None,
    ) -> None:
        self._gallery = gallery
        self._cfg: MultiViewConfig = multiview_cfg or MultiViewConfig()
        self._builder = MultiViewBuilder(self._cfg)

        # person_id -> MultiViewPersonModel
        self._models: Dict[str, MultiViewPersonModel] = {}

        # Dirty flag: set to True whenever gallery changes (enroll, add
        # template, delete person, load/reset). Higher layers are
        # responsible for calling mark_dirty() in those situations.
        self._dirty: bool = True

        # Optional: simple snapshot of how many persons/templates we last saw.
        # This is just for logging/diagnostics; we do not rely on it for
        # correctness.
        self._last_person_count: int = 0
        self._last_template_count: int = 0

    # ------------------------------------------------------------------ #
    # Dirty / refresh logic                                              #
    # ------------------------------------------------------------------ #

    def mark_dirty(self) -> None:
        """
        Mark multi-view models as stale.

        Call this whenever the underlying gallery changes in a way that
        affects templates:

          - new person enrolled
          - templates added/removed
          - gallery reset / load from disk
        """
        self._dirty = True

    def refresh_if_needed(self) -> None:
        """
        Rebuild all MultiViewPersonModel objects if they are marked dirty.

        Safe to call every frame; it will be a no-op if nothing changed.
        """
        if not self._dirty:
            return

        persons = self._gallery.persons  # type: ignore[attr-defined]
        total_templates = 0
        models: Dict[str, MultiViewPersonModel] = {}

        for pid, entry in persons.items():
            if not entry.templates:
                continue

            # Convert FaceTemplate objects into MultiViewSample list using
            # the builder helper. We let the builder interpret yaw/pitch/
            # quality and any guided pose labels from metadata.
            samples = self._builder.samples_from_templates(
                entry.templates,
                source="gallery_template",
            )
            if not samples:
                continue

            total_templates += len(samples)

            try:
                model = self._builder.build_person_model(pid, samples)
            except Exception as exc:
                # We log and skip this person instead of failing the whole
                # refresh; classic pipeline can still use flat templates.
                logger.exception(
                    "Failed to build MultiViewPersonModel for %s: %s", pid, exc
                )
                continue

            models[pid] = model

        self._models = models
        self._dirty = False

        self._last_person_count = len(persons)
        self._last_template_count = total_templates

        logger.info(
            "MultiViewGalleryView refreshed: persons=%d, mv_models=%d, "
            "templates_seen=%d",
            len(persons),
            len(models),
            total_templates,
        )

    # ------------------------------------------------------------------ #
    # Accessors                                                          #
    # ------------------------------------------------------------------ #

    def get_person_model(self, person_id: str) -> Optional[MultiViewPersonModel]:
        """
        Return the MultiViewPersonModel for a given person_id, or None if
        the person has no multi-view model (no templates, or build error).
        """
        self.refresh_if_needed()
        return self._models.get(person_id)

    def get_all_models(self) -> Dict[str, MultiViewPersonModel]:
        """
        Return a shallow copy of all person_id -> MultiViewPersonModel
        mappings.

        Useful for identity/multiview_matcher or for offline evaluation.
        """
        self.refresh_if_needed()
        return dict(self._models)

    def iter_models(self) -> Iterable[Tuple[str, MultiViewPersonModel]]:
        """
        Iterate over (person_id, MultiViewPersonModel) pairs.

        This avoids copying the whole dict for streaming use cases.
        """
        self.refresh_if_needed()
        return self._models.items()

    @property
    def models(self) -> Dict[str, MultiViewPersonModel]:
        """
        Read-only view of built multi-view models.

        This keeps existing tools (e.g. MultiViewDiagnostics, mv_report.py)
        working even though the internal attribute is _models.
        """
        self.refresh_if_needed()
        return self._models

    # ------------------------------------------------------------------ #
    # Diagnostics / summaries                                            #
    # ------------------------------------------------------------------ #

    def person_summary(self, person_id: str) -> Optional[PersonMultiViewSummary]:
        """
        Return a compact summary of one person's multi-view coverage, or
        None if they have no model.
        """
        model = self.get_person_model(person_id)
        if model is None:
            return None

        present_bins = [
            bin_name
            for bin_name, bin_state in model.bins.items()
            if bin_state.num_samples > 0
        ]
        num_bins_filled = len(present_bins)
        total_samples = int(model.total_samples)

        if total_samples > 0:
            avg_quality = float(model.quality_sum / max(1, model.quality_count))
        else:
            avg_quality = 0.0

        return PersonMultiViewSummary(
            person_id=person_id,
            num_bins_filled=num_bins_filled,
            total_samples=total_samples,
            avg_quality=avg_quality,
            present_bins=present_bins,
        )

    def all_summaries(self) -> List[PersonMultiViewSummary]:
        """
        Return multi-view coverage summaries for all persons that have
        models. This is very useful for CLI / reports / unit tests.
        """
        self.refresh_if_needed()
        out: List[PersonMultiViewSummary] = []
        for pid in sorted(self._models.keys()):
            summary = self.person_summary(pid)
            if summary is not None:
                out.append(summary)
        return out

    def debug_print_overview(self) -> None:
        """
        Log a compact overview of multi-view coverage across the gallery.

        This is intended for debugging and for your biometric course
        demonstrations (you can show that your system really stores a
        structured 3D-like head per person).
        """
        self.refresh_if_needed()

        if not self._models:
            logger.info("MultiViewGalleryView: no models available.")
            return

        summaries = self.all_summaries()
        total_persons = len(summaries)
        if total_persons == 0:
            logger.info("MultiViewGalleryView: summaries empty.")
            return

        filled_bins_counts = [s.num_bins_filled for s in summaries]
        total_samples = sum(s.total_samples for s in summaries)

        avg_bins_filled = float(np.mean(filled_bins_counts))
        max_bins_filled = max(filled_bins_counts)
        min_bins_filled = min(filled_bins_counts)

        logger.info(
            "MultiView overview: persons=%d, total_samples=%d, "
            "avg_bins_filled=%.2f, min_bins_filled=%d, max_bins_filled=%d",
            total_persons,
            total_samples,
            avg_bins_filled,
            min_bins_filled,
            max_bins_filled,
        )

        # For deeper debugging you can uncomment this to log each person:
        # for s in summaries:
        #     logger.info(
        #         "  %s: bins=%d (%s), samples=%d, avg_quality=%.3f",
        #         s.person_id,
        #         s.num_bins_filled,
        #         ",".join(s.present_bins),
        #         s.total_samples,
        #         s.avg_quality,
        #     )
