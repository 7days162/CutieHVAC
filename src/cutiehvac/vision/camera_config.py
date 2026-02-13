from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """
    USB 카메라 전용 설정 클래스.

    Attributes:
        index: USB 카메라 인덱스 (보통 0)
        width: 요청 캡처 해상도 가로 픽셀 (카메라/드라이버에 따라 적용 안 될 수도 있음)
        height: 요청 캡처 해상도 세로 픽셀
        fps: 요청 FPS (카메라에 따라 적용 안 될 수도 있음)
        buffer_size: OpenCV 내부 버퍼 크기 (지연 최소화를 위해 1 권장)
        resize_to: 캡처 후 프레임을 (w, h)로 리사이즈할 경우 사용
        flip_code: OpenCV flip 코드 (-1, 0, 1)
        rotate_deg: 90/180/270 도 회전 (시계방향)
        name: 로그/디버깅용 카메라 이름
    """

    # USB 카메라 번호 (0이 기본 카메라)
    index: int = 0

    # 캡처 설정
    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: Optional[float] = None

    # OpenCV 옵션
    buffer_size: Optional[int] = 1

    # 후처리 옵션
    resize_to: Optional[Tuple[int, int]] = None
    flip_code: Optional[int] = None
    rotate_deg: Optional[int] = None  # 90, 180, 270

    # 이름
    name: str = "usb_camera"

    def __post_init__(self) -> None:
        """설정값 유효성 검사"""

        if self.index < 0:
            raise ValueError("카메라 인덱스는 0 이상이어야 합니다.")

        if self.width is not None and self.width <= 0:
            raise ValueError("width는 양수 또는 None이어야 합니다.")

        if self.height is not None and self.height <= 0:
            raise ValueError("height는 양수 또는 None이어야 합니다.")

        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps는 양수 또는 None이어야 합니다.")

        if self.buffer_size is not None and self.buffer_size < 0:
            raise ValueError("buffer_size는 0 이상 또는 None이어야 합니다.")

        if self.resize_to is not None:
            w, h = self.resize_to
            if w <= 0 or h <= 0:
                raise ValueError("resize_to는 (양수, 양수) 형태여야 합니다.")

        if self.rotate_deg is not None and self.rotate_deg not in (90, 180, 270):
            raise ValueError("rotate_deg는 90, 180, 270 중 하나이거나 None이어야 합니다.")


# USB 카메라 전용 편의 생성 함수
def make_usb_camera(index: int = 0, *, width: int = 640, height: int = 480, fps: Optional[float] = None) -> CameraConfig:
    """USB 카메라 설정을 간단히 생성하는 함수"""

    return CameraConfig(
        index=index,
        width=width,
        height=height,
        fps=fps,
        name=f"usb:{index}",
    )
