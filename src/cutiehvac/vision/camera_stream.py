from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

from .camera_config import CameraConfig


@dataclass(frozen=True, slots=True)
class FrameResult:
    """프레임 읽기 결과(성공 여부 + 프레임 + 타임스탬프)"""

    ok: bool
    frame: Optional["cv2.Mat"]
    timestamp: float


class CameraStream:
    """USB 카메라 스트림 래퍼.

    - OpenCV VideoCapture를 감싼 클래스
    - 설정(CameraConfig) 기반으로 캡처 옵션을 best-effort로 적용
    - read()는 (ok, frame) 대신 FrameResult를 반환해서 디버깅/로깅에 유리
    """

    def __init__(self, config: CameraConfig):
        self._cfg = config
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def config(self) -> CameraConfig:
        return self._cfg

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        """카메라를 연다. 이미 열려 있으면 그대로 둔다."""

        if self.is_opened():
            return

        self._cap = cv2.VideoCapture(self._cfg.index)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"USB 카메라를 열 수 없습니다. index={self._cfg.index}")

        # 지연 최소화를 위해 버퍼 크기 최소화 (지원 안 되면 무시됨)
        if self._cfg.buffer_size is not None:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self._cfg.buffer_size))

        # 해상도/FPS는 카메라가 지원하는 범위 내에서 best-effort로 적용됨
        if self._cfg.width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._cfg.width))
        if self._cfg.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._cfg.height))
        if self._cfg.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, float(self._cfg.fps))

        # 적용값을 로그로 보고 싶으면 여기서 get()으로 확인 가능
        # 실제 적용 여부는 장치/드라이버마다 다르므로 "요청값"으로만 생각하는 게 안전

    def release(self) -> None:
        """카메라를 닫는다."""

        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def read(self) -> FrameResult:
        """프레임을 1장 읽는다."""

        if not self.is_opened():
            raise RuntimeError("카메라가 열려있지 않습니다. open()을 먼저 호출하세요.")

        ts = time.time()
        ok, frame = self._cap.read()  # type: ignore[union-attr]

        if not ok or frame is None:
            return FrameResult(ok=False, frame=None, timestamp=ts)

        frame = self._postprocess(frame)
        return FrameResult(ok=True, frame=frame, timestamp=ts)

    def get_actual_capture_props(self) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        """현재 캡처 속성(가로/세로/FPS)을 조회한다.

        카메라/드라이버가 반환하지 못하면 None이 나올 수 있다.
        """

        if not self.is_opened():
            return None, None, None

        w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # type: ignore[union-attr]
        h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type: ignore[union-attr]
        fps = self._cap.get(cv2.CAP_PROP_FPS)  # type: ignore[union-attr]

        # OpenCV가 0.0을 돌려주는 경우가 있어서 방어
        w_i = int(w) if w and w > 0 else None
        h_i = int(h) if h and h > 0 else None
        fps_f = float(fps) if fps and fps > 0 else None
        return w_i, h_i, fps_f

    def _postprocess(self, frame: "cv2.Mat") -> "cv2.Mat":
        """캡처 후 후처리(리사이즈/플립/회전)"""

        # 리사이즈
        if self._cfg.resize_to is not None:
            w, h = self._cfg.resize_to
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        # 플립
        if self._cfg.flip_code is not None:
            frame = cv2.flip(frame, self._cfg.flip_code)

        # 회전
        if self._cfg.rotate_deg is not None:
            if self._cfg.rotate_deg == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self._cfg.rotate_deg == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self._cfg.rotate_deg == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
