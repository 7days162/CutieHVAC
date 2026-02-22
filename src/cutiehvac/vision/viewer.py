from dataclasses import dataclass
from typing import Optional
import cv2


@dataclass(frozen=True)
class ViewerConfig:
    window_name: str = "CutieHVAC"
    wait_ms: int = 1
    resizable: bool = True
    always_on_top: bool = False


class FrameViewer:
    def __init__(self, config: Optional[ViewerConfig] = None):
        # 자바 스타일의 기본값 할당
        if config is None:
            config = ViewerConfig()
        self._cfg = config
        self._opened = False

    def open(self):
        if self._opened:
            return

        # 창 모드 설정
        if self._cfg.resizable:
            flags = cv2.WINDOW_NORMAL
        else:
            flags = cv2.WINDOW_AUTOSIZE

        cv2.namedWindow(self._cfg.window_name, flags)

        # 항상 위 옵션 (실패해도 무시)
        if self._cfg.always_on_top:
            try:
                cv2.setWindowProperty(self._cfg.window_name, cv2.WND_PROP_TOPMOST, 1)
            except:
                pass

        self._opened = True

    def show(self, frame) -> int:
        if not self._opened:
            self.open()

        cv2.imshow(self._cfg.window_name, frame)
        return cv2.waitKey(self._cfg.wait_ms)

    def is_quit_key(self, key_code: int) -> bool:
        # q 또는 ESC 키 확인
        k = key_code & 0xFF
        return k == ord('q') or k == 27

    def close(self):
        if not self._opened:
            return

        try:
            cv2.destroyWindow(self._cfg.window_name)
        finally:
            self._opened = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()