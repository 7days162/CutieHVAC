import time
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple
from .camera_config import CameraConfig


@dataclass(frozen=True)
class FrameResult:
    ok: bool
    frame: Optional[object]
    timestamp: float


class CameraStream:
    def __init__(self, config: CameraConfig):
        self._cfg = config
        self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self):
        if self.is_opened():
            return

        # 인덱스 타입에 따라 오픈 방식 결정
        if isinstance(self._cfg.index, str):
            self._cap = cv2.VideoCapture(self._cfg.index, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self._cfg.index)

        if not self._cap.isOpened():
            if self._cap is not None:
                self._cap.release()
            self._cap = None
            raise RuntimeError(f"카메라 연결 실패: {self._cfg.index}")

    def read(self) -> FrameResult:
        if not self.is_opened():
            return FrameResult(False, None, time.time())

        ok, frame = self._cap.read()
        return FrameResult(ok, frame, time.time())

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def get_capture_info(self):
        """현재 카메라 설정값 확인"""
        if not self.is_opened():
            return None, None, None

        w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return w, h, fps