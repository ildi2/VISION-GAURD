
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import numpy as np

from .config import FaceConfig, default_face_config

logger = logging.getLogger(__name__)


class FaceEmbedder:

    def __init__(self, cfg: Optional[FaceConfig] = None) -> None:
        self.cfg = cfg or default_face_config()
        self._dim = int(self.cfg.gallery.dim)

        logger.info(
            "FaceEmbedder initialised in passthrough mode "
            "(expects precomputed embeddings, dim=%d).",
            self._dim,
        )


    @property
    def dim(self) -> int:
        return self._dim


    def warmup(self) -> None:
        return

    def _handle_dim_mismatch(self, observed_dim: int) -> None:
        if observed_dim == self._dim:
            return

        logger.warning(
            "FaceEmbedder received embedding with dim=%d, expected %d. "
            "Updating internal dim to %d. Make sure gallery/config are "
            "consistent with the embedding backend.",
            observed_dim,
            self._dim,
            observed_dim,
        )
        self._dim = observed_dim

    def _ensure_embedding_1d(self, vector: np.ndarray) -> np.ndarray:
        emb = np.asarray(vector, dtype=np.float32).reshape(-1)

        if emb.size != self._dim:
            self._handle_dim_mismatch(emb.size)

        norm = float(np.linalg.norm(emb))
        if norm > 1e-6:
            emb /= norm
        else:
            emb[:] = 0.0

        return emb


    def embed(self, vector_or_image: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector_or_image)

        if arr.ndim == 3 and arr.shape[2] == 3:
            raise RuntimeError(
                "FaceEmbedder.embed no longer supports raw image inputs. "
                "Embeddings must be produced by FaceDetectorAligner "
                "(buffalo_l) and then passed here only if you need "
                "normalisation."
            )

        return self._ensure_embedding_1d(arr)

    def embed_many(self, vectors: Iterable[np.ndarray]) -> np.ndarray:
        embs: List[np.ndarray] = []

        for idx, v in enumerate(vectors):
            try:
                embs.append(self.embed(v))
            except Exception as exc:
                logger.warning(
                    "Failed to normalise one embedding in embed_many (index=%d): %s",
                    idx,
                    exc,
                )
                embs.append(np.zeros(self._dim, dtype=np.float32))

        if not embs:
            return np.zeros((0, self._dim), dtype=np.float32)

        return np.stack(embs, axis=0)


    def is_compatible_dim(self, vector: np.ndarray) -> bool:
        arr = np.asarray(vector).reshape(-1)
        return int(arr.size) == int(self._dim)
