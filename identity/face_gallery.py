
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from face.config import FaceGalleryConfig, default_face_config
from identity.crypto import (
    CryptoConfig,
    encrypt_json,
    decrypt_json,
    CryptoError,
    CryptoKeyError,
    CryptoCiphertextError,
    CryptoVersionError,
    get_cached_key_fingerprint,
)

from face.multiview_types import (
    MultiViewConfig,
    MultiViewPersonModel,
)
from face.multiview_builder import MultiViewBuilder

logger = logging.getLogger(__name__)


try:
    import faiss
except Exception as exc:
    faiss = None
    _faiss_error = exc
else:
    _faiss_error = None


Category = Literal["resident", "visitor", "watchlist", "unknown"]


@dataclass
class FaceTemplate:

    embedding: np.ndarray
    condition: str = "neutral"
    created_at: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonEntry:

    person_id: str
    category: Category = "resident"
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    templates: List[FaceTemplate] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())


@dataclass
class PersonSummary:

    person_id: str
    category: Category
    name: Optional[str]
    num_templates: int


@dataclass
class SearchResult:

    person_id: str
    distance: float
    score: float
    condition: str
    category: Category
    name: Optional[str]


class FaceGallery:

    def __init__(
        self,
        cfg: Optional[FaceGalleryConfig] = None,
        crypto_cfg: Optional[CryptoConfig] = None,
        multiview_cfg: Optional[MultiViewConfig] = None,
    ) -> None:
        if cfg is None:
            cfg = default_face_config().gallery
        self.cfg = cfg
        self.crypto_cfg = crypto_cfg or CryptoConfig()

        self._multiview_cfg: MultiViewConfig = multiview_cfg or MultiViewConfig()
        self._mv_builder: MultiViewBuilder = MultiViewBuilder(self._multiview_cfg)

        self._path: Optional[Path] = (
            Path(self.cfg.gallery_path) if self.cfg.gallery_path else None
        )

        self._persons: Dict[str, PersonEntry] = {}
        self._next_person_index: int = 1

        self._embeddings: Optional[np.ndarray] = None
        self._index_meta: List[Tuple[str, int]] = []
        self._faiss_index: Any = None
        self._index_dirty: bool = True

        self._mv_models: Dict[str, MultiViewPersonModel] = {}
        self._mv_dirty: bool = True

        self._locked_due_to_load_error: bool = False
        self._last_load_error: Optional[str] = None

        if self._path is not None:
            self.load_if_exists()


    def enroll_person(
        self,
        templates: Optional[Sequence[Union[np.ndarray, FaceTemplate]]] = None,
        *,
        conditions: Optional[Sequence[str]] = None,
        category: Category = "resident",
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        person_id: Optional[str] = None,
        surname: Optional[str] = None,
        embeddings: Optional[np.ndarray] = None,
        condition: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        if templates is None and embeddings is not None:
            arr = np.asarray(embeddings, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError(
                    f"`embeddings` must be 1D or 2D array, got shape {arr.shape}"
                )
            templates = [row.copy() for row in arr]

            if conditions is None and condition is not None:
                conditions = [condition] * len(templates)

            base_meta = dict(metadata or {})
            if notes:
                base_meta.setdefault("notes", notes)
            metadata = base_meta

        if not templates:
            raise ValueError("enroll_person requires at least one template embedding.")

        if person_id is None:
            person_id = self._allocate_person_id()
        else:
            if person_id in self._persons:
                logger.warning(
                    "enroll_person called with existing person_id '%s'. "
                    "New templates will be appended to this entry.",
                    person_id,
                )

        person_meta = dict(metadata or {})
        if surname:
            person_meta.setdefault("surname", surname)

        if person_id in self._persons:
            entry = self._persons[person_id]
            if category is not None:
                entry.category = category
            if name is not None:
                entry.name = name
            entry.metadata.update(person_meta)
        else:
            entry = PersonEntry(
                person_id=person_id,
                category=category,
                name=name,
                metadata=person_meta,
            )
            self._persons[person_id] = entry

        if isinstance(templates[0], FaceTemplate):
            for tmpl in templates:
                tmpl.embedding = self._validate_embedding(tmpl.embedding)
                self._normalise_template_metadata(tmpl)
                entry.templates.append(tmpl)

        else:
            if conditions is None:
                conditions = ["neutral"] * len(templates)
            if len(conditions) != len(templates):
                raise ValueError("conditions length must match templates length.")

            for emb_vec, cond in zip(templates, conditions):
                emb_valid = self._validate_embedding(emb_vec)
                tmpl = FaceTemplate(
                    embedding=emb_valid,
                    condition=cond,
                    metadata={},
                )
                self._normalise_template_metadata(tmpl)
                entry.templates.append(tmpl)

        self._index_dirty = True
        self._mv_dirty = True

        logger.info(
            "Enrolled person %s (category=%s, templates_total=%d).",
            person_id,
            entry.category,
            len(entry.templates),
        )
        return person_id

    def add_template(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        condition: str = "neutral",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if person_id not in self._persons:
            logger.warning(
                "add_template called for unknown person_id '%s' – creating "
                "a new PersonEntry with default category='resident'.",
                person_id,
            )
            self._persons[person_id] = PersonEntry(person_id=person_id)

        emb = self._validate_embedding(embedding)
        tmpl = FaceTemplate(
            embedding=emb,
            condition=condition,
            metadata=metadata or {},
        )
        self._normalise_template_metadata(tmpl)
        self._persons[person_id].templates.append(tmpl)

        self._index_dirty = True
        self._mv_dirty = True

    def delete_person(self, person_id: str) -> bool:
        if person_id in self._persons:
            del self._persons[person_id]
            self._index_dirty = True
            self._mv_dirty = True
            self._mv_models.pop(person_id, None)
            return True
        return False

    def update_metadata(
        self,
        person_id: str,
        *,
        category: Optional[Category] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = self._persons.get(person_id)
        if p is None:
            raise KeyError(f"Person '{person_id}' not found.")
        if category is not None:
            p.category = category
        if name is not None:
            p.name = name
        if metadata is not None:
            p.metadata.update(metadata)


    def search_best(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> Optional[SearchResult]:
        emb = self._validate_embedding(embedding, normalize=True)
        self._ensure_index()

        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return None

        k = max(1, min(k, self._embeddings.shape[0]))

        if self._faiss_index is not None:
            q = emb.reshape(1, -1)
            if self.cfg.metric == "cosine":
                sims, idxs = self._faiss_index.search(q, k)
                sims = sims[0]
                idxs = idxs[0]
                best_idx = int(idxs[0])
                sim = float(sims[0])
                distance = 1.0 - sim
            else:
                dists, idxs = self._faiss_index.search(q, k)
                dists = dists[0]
                idxs = idxs[0]
                best_idx = int(idxs[0])
                distance = float(dists[0])
        else:
            E = self._embeddings
            if self.cfg.metric == "cosine":
                sims = (E @ emb.reshape(-1, 1)).reshape(-1)
                best_idx = int(np.argmax(sims))
                sim = float(sims[best_idx])
                distance = 1.0 - sim
            else:
                diffs = E - emb.reshape(1, -1)
                dists = np.sum(diffs * diffs, axis=1)
                best_idx = int(np.argmin(dists))
                distance = float(dists[best_idx])

        person_id, tmpl_idx = self._index_meta[best_idx]
        person = self._persons[person_id]
        tmpl = person.templates[tmpl_idx]

        if self.cfg.metric == "cosine":
            score = max(0.0, 1.0 - distance)
        else:
            score = 1.0 / (1.0 + distance)

        return SearchResult(
            person_id=person_id,
            distance=distance,
            score=score,
            condition=tmpl.condition,
            category=person.category,
            name=person.name,
        )


    def _ensure_multiview_models(self) -> None:
        if not self._mv_dirty:
            return

        self._mv_models.clear()

        if not self._persons:
            self._mv_dirty = False
            return

        for pid, entry in self._persons.items():
            if not entry.templates:
                continue

            samples = self._mv_builder.samples_from_templates(
                entry.templates,
                source="gallery_template",
            )
            try:
                mv_model = self._mv_builder.build_person_model(pid, samples)
                self._mv_models[pid] = mv_model
            except Exception as exc:
                logger.exception(
                    "Failed to build multi-view model for %s: %s", pid, exc
                )

        self._mv_dirty = False
        logger.info(
            "Multi-view models built for %d persons.", len(self._mv_models)
        )

    def get_multiview_model(self, person_id: str) -> Optional[MultiViewPersonModel]:
        self._ensure_multiview_models()
        return self._mv_models.get(person_id)

    def get_all_multiview_models(self) -> Dict[str, MultiViewPersonModel]:
        self._ensure_multiview_models()
        return dict(self._mv_models)

    def rebuild_multiview_models(self) -> None:
        self._mv_dirty = True
        self._ensure_multiview_models()


    @property
    def locked_due_to_error(self) -> bool:
        return self._locked_due_to_load_error

    @property
    def last_load_error(self) -> Optional[str]:
        return self._last_load_error

    @property
    def multiview_cfg(self) -> MultiViewConfig:
        return self._multiview_cfg

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self._path or self.cfg.gallery_path
        path = Path(path)

        if self._locked_due_to_load_error:
            raise RuntimeError(
                f"FaceGallery at '{path}' is locked due to a previous load error "
                f"({self._last_load_error}). "
                "Refusing to overwrite existing file. "
                "If you really want to reset the gallery, delete the file "
                "explicitly (or use a 'reset-gallery' command) and restart."
            )

        data = self._to_serializable()
        blob = encrypt_json(
            data,
            env_var=self.crypto_cfg.env_var,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            f.write(blob)
        os.replace(tmp_path, path)
        logger.info("Face gallery saved to %s (persons=%d).", path, len(self._persons))

    def load(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self._path or self.cfg.gallery_path
        path = Path(path)

        with path.open("rb") as f:
            blob = f.read()

        data = decrypt_json(
            blob,
            env_var=self.crypto_cfg.env_var,
        )
        self._from_serializable(data)
        self._index_dirty = True
        self._mv_dirty = True
        logger.info("Face gallery loaded from %s (persons=%d).", path, len(self._persons))

    def load_if_exists(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = self._path or self.cfg.gallery_path
        path = Path(path)
        if not path.exists():
            return

        try:
            self.load(path)
            self._locked_due_to_load_error = False
            self._last_load_error = None
        except (CryptoKeyError, CryptoCiphertextError, CryptoVersionError) as exc:
            fp = get_cached_key_fingerprint(self.crypto_cfg.env_var) or "unknown"
            self._locked_due_to_load_error = True
            self._last_load_error = str(exc)
            logger.error(
                "Failed to decrypt face gallery at '%s': %s "
                "(env_var=%s, key_fp=%s). The on-disk file was NOT modified. "
                "Most likely causes: the encryption key changed after the "
                "gallery was created, or the file is corrupted. "
                "This gallery instance is now locked; it will not overwrite "
                "the file until you delete/reset it explicitly.",
                path,
                exc,
                self.crypto_cfg.env_var,
                fp,
            )
            self._persons.clear()
            self._embeddings = None
            self._index_meta = []
            self._faiss_index = None
            self._index_dirty = False
            self._mv_models.clear()
            self._mv_dirty = False
        except CryptoError as exc:
            self._locked_due_to_load_error = True
            self._last_load_error = str(exc)
            logger.error(
                "CryptoError while loading face gallery from '%s': %s. "
                "The gallery is locked for this process.",
                path,
                exc,
            )
            self._persons.clear()
            self._embeddings = None
            self._index_meta = []
            self._faiss_index = None
            self._index_dirty = False
            self._mv_models.clear()
            self._mv_dirty = False
        except Exception as exc:
            self._last_load_error = str(exc)
            logger.error(
                "Failed to load face gallery from %s (non-crypto error): %s",
                path,
                exc,
            )


    @property
    def persons(self) -> Dict[str, PersonEntry]:
        return self._persons

    def get_person(self, person_id: str) -> Optional[PersonEntry]:
        return self._persons.get(person_id)

    def list_persons(self) -> List[PersonSummary]:
        summaries: List[PersonSummary] = []
        for person_id, entry in self._persons.items():
            summaries.append(
                PersonSummary(
                    person_id=person_id,
                    category=entry.category,
                    name=entry.name,
                    num_templates=len(entry.templates),
                )
            )
        return summaries


    def _allocate_person_id(self) -> str:
        while True:
            pid = f"p_{self._next_person_index:04d}"
            self._next_person_index += 1
            if pid not in self._persons:
                return pid

    def _validate_embedding(
        self,
        embedding: np.ndarray,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.size != self.cfg.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.cfg.dim}, got {emb.size}."
            )
        if normalize:
            norm = float(np.linalg.norm(emb))
            if norm > 1e-6:
                emb /= norm
            else:
                emb[:] = 0.0
        return emb

    def _normalise_template_metadata(self, tmpl: FaceTemplate) -> None:
        m = tmpl.metadata
        if not isinstance(m, dict):
            return

        if "pose_bin_hint" not in m:
            if "pose_bin" in m and isinstance(m["pose_bin"], str):
                m["pose_bin_hint"] = str(m["pose_bin"]).upper()
        else:
            if isinstance(m["pose_bin_hint"], str):
                m["pose_bin_hint"] = m["pose_bin_hint"].upper()

        if "yaw_deg" not in m and "yaw" in m:
            try:
                m["yaw_deg"] = float(m["yaw"])
            except Exception:
                pass
        if "pitch_deg" not in m and "pitch" in m:
            try:
                m["pitch_deg"] = float(m["pitch"])
            except Exception:
                pass

        if "quality" in m:
            try:
                m["quality"] = float(m["quality"])
            except Exception:
                pass

        if "timestamp" not in m and "ts" in m:
            try:
                m["timestamp"] = float(m["ts"])
            except Exception:
                pass


    def _ensure_index(self) -> None:
        if not self._index_dirty:
            return
        self._rebuild_index()
        self._index_dirty = False

    def _rebuild_index(self) -> None:
        all_embs: List[np.ndarray] = []
        meta: List[Tuple[str, int]] = []
        for person_id, person in self._persons.items():
            for idx, tmpl in enumerate(person.templates):
                all_embs.append(self._validate_embedding(tmpl.embedding, normalize=True))
                meta.append((person_id, idx))

        if not all_embs:
            self._embeddings = None
            self._index_meta = []
            self._faiss_index = None
            return

        self._embeddings = np.stack(all_embs, axis=0)
        self._index_meta = meta

        if faiss is None:
            if _faiss_error is not None:
                logger.warning(
                    "FAISS not available (%s). Using NumPy search for face gallery.",
                    _faiss_error,
                )
            self._faiss_index = None
            return

        dim = self.cfg.dim
        if self.cfg.metric == "cosine":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(self._embeddings.astype(np.float32))
        self._faiss_index = index
        logger.info(
            "Face gallery FAISS index rebuilt (templates=%d).",
            self._embeddings.shape[0],
        )


    def _to_serializable(self) -> Dict[str, Any]:
        persons_data: List[Dict[str, Any]] = []
        max_pid = 0

        for p in self._persons.values():
            try:
                n = int(p.person_id.split("_")[-1])
                max_pid = max(max_pid, n)
            except Exception:
                pass

            t_list: List[Dict[str, Any]] = []
            for t in p.templates:
                t_list.append(
                    {
                        "condition": t.condition,
                        "created_at": t.created_at,
                        "metadata": t.metadata,
                        "embedding": t.embedding.astype(float).tolist(),
                    }
                )

            persons_data.append(
                {
                    "person_id": p.person_id,
                    "category": p.category,
                    "name": p.name,
                    "metadata": p.metadata,
                    "templates": t_list,
                    "created_at": getattr(p, "created_at", time.time()),
                }
            )

        self._next_person_index = max(max_pid + 1, self._next_person_index)

        return {
            "version": 1,
            "dim": self.cfg.dim,
            "metric": self.cfg.metric,
            "persons": persons_data,
        }

    def _from_serializable(self, data: Dict[str, Any]) -> None:
        if data.get("dim") != self.cfg.dim:
            logger.warning(
                "Loaded gallery dim=%s differs from config dim=%s.",
                data.get("dim"),
                self.cfg.dim,
            )

        self._persons.clear()
        persons_data = data.get("persons", []) or []

        max_pid = 0

        for p_data in persons_data:
            pid = str(p_data["person_id"])
            entry = PersonEntry(
                person_id=pid,
                category=p_data.get("category", "resident"),
                name=p_data.get("name"),
                metadata=p_data.get("metadata", {}) or {},
                created_at=float(p_data.get("created_at", time.time())),
            )

            t_list = p_data.get("templates", []) or []
            for t_data in t_list:
                emb = np.asarray(t_data["embedding"], dtype=np.float32).reshape(-1)
                tmpl = FaceTemplate(
                    embedding=emb,
                    condition=t_data.get("condition", "neutral"),
                    created_at=float(t_data.get("created_at", time.time())),
                    metadata=t_data.get("metadata", {}) or {},
                )
                self._normalise_template_metadata(tmpl)
                entry.templates.append(tmpl)

            self._persons[pid] = entry

            try:
                n = int(pid.split("_")[-1])
                max_pid = max(max_pid, n)
            except Exception:
                pass

        self._next_person_index = max(max_pid + 1, 1)
        self._index_dirty = True
        self._mv_dirty = True
