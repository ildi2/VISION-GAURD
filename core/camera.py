
from __future__ import annotations

import threading
import queue
from typing import Optional

import cv2


def open_camera(index: int = 0, w: int = 640, h: int = 480, fps: int = 30,
                mjpeg: bool = True, buffersize: int = 1) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)

    return cap


class CameraSource:

    def __init__(self, cam_index: int = 0, **kw) -> None:
        self.cap = open_camera(cam_index, **kw)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not available")

        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._running: bool = False

    def _producer(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue

            if not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass

            self._q.put(frame)

    def start(self) -> None:
        self._running = True
        t = threading.Thread(target=self._producer, daemon=True)
        t.start()

    def read_latest(self, timeout: float = 1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        self.cap.release()
