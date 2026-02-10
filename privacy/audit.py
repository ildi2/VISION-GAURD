
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("privacy.audit")


class AuditWriter:
    
    def __init__(
        self,
        output_dir: str,
        filename: str = "privacy_audit.jsonl",
        flush_interval_sec: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._output_dir = Path(output_dir)
        self._filename = filename
        self._flush_interval_sec = flush_interval_sec
        self._file_handle: Optional[Any] = None
        self._last_flush_ts: float = 0.0
        self._entries_since_flush: int = 0
        self._total_entries: int = 0
        self._file_path: Optional[Path] = None
        
        if self._enabled:
            self._open_file()
    
    def _open_file(self) -> None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            
            self._file_path = self._output_dir / self._filename
            
            self._file_handle = open(self._file_path, "a", encoding="utf-8")
            self._last_flush_ts = time.perf_counter()
            
            log.info("Audit log opened: %s", self._file_path)
            
        except Exception as e:
            log.error("Failed to open audit file: %s", e)
            self._file_handle = None
            self._enabled = False
    
    def write_entry(self, entry: Dict[str, Any]) -> bool:
        if not self._enabled or self._file_handle is None:
            return False
        
        try:
            line = json.dumps(entry, separators=(",", ":"), default=str)
            self._file_handle.write(line + "\n")
            
            self._entries_since_flush += 1
            self._total_entries += 1
            
            self._maybe_flush()
            
            return True
            
        except Exception as e:
            log.error("Failed to write audit entry: %s", e)
            return False
    
    def _maybe_flush(self) -> None:
        if self._file_handle is None:
            return
        
        now = time.perf_counter()
        elapsed = now - self._last_flush_ts
        
        if self._flush_interval_sec <= 0 or elapsed >= self._flush_interval_sec:
            try:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
                self._last_flush_ts = now
                self._entries_since_flush = 0
            except Exception as e:
                log.warning("Failed to flush audit log: %s", e)
    
    def flush(self) -> None:
        if self._file_handle is not None:
            try:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
                self._last_flush_ts = time.perf_counter()
                self._entries_since_flush = 0
            except Exception as e:
                log.warning("Failed to force flush audit log: %s", e)
    
    def close(self) -> None:
        if self._file_handle is not None:
            try:
                self._file_handle.flush()
                self._file_handle.close()
                log.info(
                    "Audit log closed: %s (total entries: %d)",
                    self._file_path,
                    self._total_entries,
                )
            except Exception as e:
                log.error("Error closing audit file: %s", e)
            finally:
                self._file_handle = None
    
    @property
    def file_path(self) -> Optional[Path]:
        return self._file_path
    
    @property
    def total_entries(self) -> int:
        return self._total_entries
    
    @property
    def is_open(self) -> bool:
        return self._file_handle is not None and self._enabled


def build_audit_entry(
    frame_ts: float,
    frame_id: int,
    emit_ts: float,
    track_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    redacted_count = sum(
        1 for t in track_entries 
        if t.get("policy_state") not in ("VISIBLE", "M1_PLACEHOLDER")
    )
    visible_count = len(track_entries) - redacted_count
    
    return {
        "ts": frame_ts,
        "frame_id": frame_id,
        "emit_ts": emit_ts,
        "tracks": track_entries,
        "redacted_count": redacted_count,
        "visible_count": visible_count,
    }


def build_track_audit_entry(
    track_id: int,
    policy_state: str,
    redaction_method: str,
    identity_id: Optional[str],
    id_source: str,
    authorized_signal: bool,
    decision_category: str,
    decision_binding_state: Optional[str],
    is_redacted: bool = False,
    lock_age_sec: Optional[float] = None,
    grace_remaining_sec: Optional[float] = None,
    transition_count: int = 0,
    mask_used: bool = False,
    mask_source: Optional[str] = None,
    mask_quality: Optional[float] = None,
    fallback_to_bbox: bool = False,
    stable_mask_used: bool = False,
    stabilization_method: Optional[str] = None,
    shrink_detected: bool = False,
    ttl_reuse: bool = False,
    mask_area: Optional[int] = None,
    stable_mask_area: Optional[int] = None,
    bbox: Optional[List[float]] = None,
) -> Dict[str, Any]:
    entry = {
        "track_id": track_id,
        "policy_state": policy_state,
        "redaction_method": redaction_method,
        "identity_id": identity_id,
        "id_source": id_source,
        "authorized_signal": authorized_signal,
        "decision_category": decision_category,
        "decision_binding_state": decision_binding_state,
        "is_redacted": is_redacted,
    }
    
    if bbox is not None:
        entry["bbox"] = list(bbox) if not isinstance(bbox, list) else bbox
    
    if lock_age_sec is not None:
        entry["lock_age_sec"] = lock_age_sec
    if grace_remaining_sec is not None:
        entry["grace_remaining_sec"] = grace_remaining_sec
    if transition_count > 0:
        entry["transition_count"] = transition_count
    
    if mask_used or fallback_to_bbox:
        entry["mask_used"] = mask_used
        if mask_source is not None:
            entry["mask_source"] = mask_source
        if mask_quality is not None:
            entry["mask_quality"] = round(mask_quality, 3)
        entry["fallback_to_bbox"] = fallback_to_bbox
    
    if stable_mask_used or ttl_reuse or shrink_detected:
        entry["stable_mask_used"] = stable_mask_used
        if stabilization_method is not None:
            entry["stabilization_method"] = stabilization_method
        entry["shrink_detected"] = shrink_detected
        entry["ttl_reuse"] = ttl_reuse
        if mask_area is not None:
            entry["mask_area"] = mask_area
        if stable_mask_area is not None:
            entry["stable_mask_area"] = stable_mask_area
    
    return entry
