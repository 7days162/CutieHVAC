from cutiehvac.vision.camera_config import make_usb_camera
from cutiehvac.vision.camera_stream import CameraStream
from cutiehvac.vision.viewer import FrameViewer

import cv2

def find_available_cameras(max_index: int = 6):
    print("=== 사용 가능한 카메라 인덱스 검색 ===")
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[OK] index {i}")
            available.append(i)
        cap.release()
    print("===================================")
    return available


def main():
    cams = find_available_cameras()

    if not cams:
        print("사용 가능한 카메라가 없습니다.")
        return

    selected_index = cams[0]
    print(f"선택된 카메라 index: {selected_index}")

    cfg = make_usb_camera(index=selected_index, width=640, height=480)

    with CameraStream(cfg) as cam, FrameViewer() as viewer:
        print("카메라 프리뷰 시작 (q 또는 ESC로 종료)")

        while True:
            result = cam.read()

            if not result.ok:
                continue

            key = viewer.show(result.frame)
            if viewer.should_quit(key):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()