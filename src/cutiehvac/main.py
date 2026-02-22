import sys
import time
from pathlib import Path

from cutiehvac.vision.camera_config import make_camera
from cutiehvac.vision.camera_stream import CameraStream
from cutiehvac.vision.detector_yolo import YoloDetector
from cutiehvac.control.hvac_controller import HVACController

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

def main():
    try:
        conf_input = input("> 탐지 임계값을 설정하세요 (0.1 ~ 1.0, 기본 0.4): ")
        conf_val = float(conf_input) if conf_input else 0.4
    except ValueError:
        conf_val = 0.4
        print("잘못된 입력입니다. 기본값(0.4)으로 진행합니다.")

    # 2. 시스템 구성 (init)
    # 제어 로직 담당
    controller = HVACController()

    # 모델 경로 (라즈베리파이5 에 최적화됨)
    MODEL_PATH = PROJECT_ROOT / "data" / "models" / "yolo_doll_v1" / "weights" / "best.pt"
    detector = YoloDetector(str(MODEL_PATH), conf_threshold=conf_val)

    # CSI 카메라 파이프라인
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
    cfg = make_camera(index=CSI_PIPELINE)

    # 3. 메인 프로세스 실행
    print("\n[INFO] 장치 초기화 및 카메라 연결 중...")

    with CameraStream(cfg) as cam:
        print("> 시스템이 정상 가동 중입니다.")
        print("> 종료하려면 'Ctrl + C'를 누르세요.\n")

        last_log_time = time.time()

        try:
            while True:
                res = cam.read()
                if not res.ok:
                    continue

                # 비전 분석
                doll_count, _ = detector.detect(res.frame)

                # 컨트롤러 업데이트
                mode, speed = controller.update(doll_count)

                if time.time() - last_log_time >= 1.0:
                    time_str = time.strftime('%H:%M:%S')

                    print(f"[{time_str}] Mode: {mode:4} | Objects: {doll_count} | Power: {speed * 100:>3.0f}%")
                    last_log_time = time.time()

        except KeyboardInterrupt:
            print("\n\n[WARN] 사용자에 의한 강제 종료 신호 감지.")
        finally:
            # 마지막 종료
            controller.shutdown()
            print("[INFO] 하드웨어 자원을 안전하게 해제했습니다.")
            print("========================================")


if __name__ == "__main__":
    main()