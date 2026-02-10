
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("privacy.writer")


CODEC_FOURCC = {
    "mp4v": "mp4v",
    "avc1": "avc1",
    "h264": "avc1",
    "xvid": "XVID",
    "mjpg": "MJPG",
}


class PrivacyWriter:
    
    def __init__(
        self,
        output_dir: str = "privacy_output",
        basename: str = "privacy_stream",
        fps: float = 30.0,
        codec: str = "mp4v",
        container: str = "mp4",
        frame_size: Optional[Tuple[int, int]] = None,
        enabled: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._basename = basename
        self._fps = fps
        self._codec = codec
        self._container = container
        self._frame_size = frame_size
        self._enabled = enabled
        
        self._writer: Optional[cv2.VideoWriter] = None
        self._file_path: Optional[Path] = None
        self._is_open = False
        self._is_failed = False
        
        self._frames_written = 0
        self._first_write_ts: Optional[float] = None
        self._last_write_ts: Optional[float] = None
        
        log.info(
            "PrivacyWriter initialized (lazy) | dir=%s, basename=%s, fps=%.1f, codec=%s, enabled=%s",
            output_dir,
            basename,
            fps,
            codec,
            enabled,
        )
    
    def set_fps(self, fps: float) -> None:
        if self._is_open:
            log.warning("Cannot change FPS after writer is already open (fps=%.1f)", self._fps)
            return
        old_fps = self._fps
        self._fps = max(1.0, min(60.0, fps))
        log.info(
            "PrivacyWriter FPS updated: %.1f -> %.1f (measured from actual frame rate)",
            old_fps, self._fps,
        )
    
    def _lazy_open(self, frame: np.ndarray) -> bool:
        if self._is_open or self._is_failed:
            return self._is_open
        
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            
            if self._frame_size is None:
                h, w = frame.shape[:2]
                self._frame_size = (w, h)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self._basename}_{timestamp}.{self._container}"
            self._file_path = self._output_dir / filename
            
            fourcc_str = CODEC_FOURCC.get(self._codec.lower(), self._codec)
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            
            self._writer = cv2.VideoWriter(
                str(self._file_path),
                fourcc,
                self._fps,
                self._frame_size,
            )
            
            if not self._writer.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {self._file_path}")
            
            self._is_open = True
            self._first_write_ts = time.time()
            
            log.info(
                "PrivacyWriter opened: %s | size=%dx%d, fps=%.1f, codec=%s",
                self._file_path,
                self._frame_size[0],
                self._frame_size[1],
                self._fps,
                self._codec,
            )
            
            return True
            
        except Exception as e:
            log.error("PrivacyWriter failed to open: %s", e)
            self._is_failed = True
            self._is_open = False
            return False
    
    def write(self, frame: np.ndarray) -> bool:
        if not self._enabled:
            return True
        
        if self._is_failed:
            return False
        
        try:
            if not self._is_open:
                if not self._lazy_open(frame):
                    return False
            
            self._writer.write(frame)
            self._frames_written += 1
            self._last_write_ts = time.time()
            
            return True
            
        except Exception as e:
            log.exception("PrivacyWriter write failed (disabling): %s", e)
            self._is_failed = True
            self._close_internal()
            return False
    
    def _close_internal(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as e:
                log.warning("Error releasing VideoWriter: %s", e)
            self._writer = None
        self._is_open = False
    
    def close(self) -> None:
        if self._is_open and self._writer is not None:
            try:
                self._writer.release()
                log.info(
                    "PrivacyWriter closed: %s | frames_written=%d",
                    self._file_path,
                    self._frames_written,
                )
            except Exception as e:
                log.warning("Error closing PrivacyWriter: %s", e)
            finally:
                self._writer = None
                self._is_open = False
        elif not self._enabled:
            log.info("PrivacyWriter closed (was disabled, never opened)")
        elif self._is_failed:
            log.info("PrivacyWriter closed (was in failed state)")
        else:
            log.info("PrivacyWriter closed (never opened, no frames emitted within delay window)")
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    @property
    def is_failed(self) -> bool:
        return self._is_failed
    
    @property
    def file_path(self) -> Optional[Path]:
        return self._file_path
    
    @property
    def frames_written(self) -> int:
        return self._frames_written
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def get_stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "is_open": self._is_open,
            "is_failed": self._is_failed,
            "file_path": str(self._file_path) if self._file_path else None,
            "frames_written": self._frames_written,
            "fps": self._fps,
            "codec": self._codec,
            "frame_size": self._frame_size,
            "first_write_ts": self._first_write_ts,
            "last_write_ts": self._last_write_ts,
        }
