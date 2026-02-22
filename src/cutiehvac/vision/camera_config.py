from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class CameraConfig:
    index: Union[int, str]
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    buffer_size: Optional[int] = None
    resize_to: Optional[Tuple[int, int]] = None
    flip_code: Optional[int] = None
    rotate_deg: Optional[int] = None
    name: str = "unnamed_camera"

    def __post_init__(self):
        if isinstance(self.index, int) and self.index < 0:
            raise ValueError(f"Invalid camera index: {self.index}")


def make_camera(
        index: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        name: str = None
) -> CameraConfig:
    if name is None:
        name = f"cam_{index}"

    return CameraConfig(
        index=index,
        width=width,
        height=height,
        fps=fps,
        name=name
    )