from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass(frozen=True, slots=True)
class ViewerConfig:
    """화면 출력(뷰어) 설정"""

    window_name: str = "CutieHVAC"

    # waitKey 지연(ms). 1이면 사실상 실시간.
    # 키 입력을 받으려면 1 이상이어야 함.
    wait_ms: int = 1

    # 창 크기 조절 허용 여부
    resizable: bool = True

    # 항상 위로(선택). OS/환경에 따라 무시될 수 있음.
    always_on_top: bool = False


class FrameViewer:
    """OpenCV로 프레임을 화면에 띄우는 클래스"""

    def __init__(self, config: Optional[ViewerConfig] = None):
        self._cfg = config or ViewerConfig()
        self._opened = False

    @property
    def config(self) -> ViewerConfig:
        return self._cfg

    def open(self) -> None:
        """창을 생성한다(한 번만)."""

        if self._opened:
            return

        flags = cv2.WINDOW_NORMAL if self._cfg.resizable else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(self._cfg.window_name, flags)

        # 항상 위로 옵션(환경에 따라 동작 안 할 수도 있음)
        if self._cfg.always_on_top:
            try:
                cv2.setWindowProperty(
                    self._cfg.window_name,
                    cv2.WND_PROP_TOPMOST,
                    1,
                )
            except Exception:
                pass

        self._opened = True

    def show(self, frame) -> int:
        """프레임을 띄우고 키 입력 코드를 반환한다.

        반환값은 cv2.waitKey() 결과(정수)이며,
        보통 `key & 0xFF` 형태로 비교한다.
        """

        if not self._opened:
            self.open()

        cv2.imshow(self._cfg.window_name, frame)
        return cv2.waitKey(self._cfg.wait_ms)

    def should_quit(self, key_code: int) -> bool:
        """종료 키(q 또는 ESC) 판단"""

        k = key_code & 0xFF
        return k == ord('q') or k == 27

    def close(self) -> None:
        """창을 닫는다."""

        if not self._opened:
            return

        try:
            cv2.destroyWindow(self._cfg.window_name)
        finally:
            self._opened = False

    def __enter__(self) -> "FrameViewer":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()