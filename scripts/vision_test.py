import sys
import time
from pathlib import Path
from gpiozero import LED

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from cutiehvac.vision.camera_config import make_camera
from cutiehvac.vision.camera_stream import CameraStream
from cutiehvac.vision.detector_yolo import YoloDetector


def main():
    # 팬 제어용 핀 설정 (GPIO 18 == 12번 핀)
    fan = LED(18)

    # 라즈베리파이 전용 카메라 파이프라인
    CSI_PIPELINE = (
        "libcamerasrc ! "
        "video/x-raw,format=YUY2,width=1536,height=864 ! "
        "videoconvert ! "
        "videoscale ! "
        "video/x-raw,width=640,height=480 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=True"
    )

    # 모델 경로 및 로드
    model_path = PROJECT_ROOT / "data" / "models" / "yolo_doll_v1" / "weights" / "best.pt"
    detector = YoloDetector(str(model_path))

    # 카메라 설정
    config = make_camera(index=CSI_PIPELINE)

    with CameraStream(config) as cam:
        print("실시간 모니터링 중... (종료하려면 Ctrl+C)")

        last_check_time = time.time()

        try:
            while True:
                result = cam.read()
                if not result.ok:
                    continue

                # 탐지
                count, _ = detector.detect(result.frame)

                # 팬 제어 로직
                if count > 0:
                    fan.on()
                    current_status = "ON"
                else:
                    fan.off()
                    current_status = "OFF"

                # 1초 간격으로 로그 출력
                now = time.time()
                if now - last_check_time >= 1.0:
                    time_str = time.strftime('%H:%M:%S')
                    print(f"[{time_str}] Fan: {current_status} | Dolls: {count}")
                    last_check_time = now

        except KeyboardInterrupt:
            print("\n사용자에 의해 종료되었습니다.")
        finally:
            fan.off()
            print("시스템을 안전하게 종료합니다.")


if __name__ == "__main__":
    main()