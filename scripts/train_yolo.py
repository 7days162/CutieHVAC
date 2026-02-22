import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

def get_latest_dataset_path(base_name):
    """가장 숫자가 높은 최신 데이터셋 폴더 찾기"""
    if not DATASETS_DIR.exists():
        return None

    latest_v = -1
    latest_path = None

    for folder in DATASETS_DIR.iterdir():
        if folder.is_dir() and folder.name.startswith(base_name + "_v"):
            try:
                # v1, v2 등에서 숫자만 추출
                v_num = int(folder.name.split("_v")[-1])
                if v_num > latest_v:
                    latest_v = v_num
                    latest_path = folder
            except:
                continue

    return latest_path

def select_device():
    """사용 가능한 최적의 장치 선택 (Mac은 mps, 아니면 cpu)"""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "0" # NVIDIA GPU
    return "cpu"

def run_training():
    # 1. 인자값 설정
    parser = argparse.ArgumentParser(description="CutieHVAC YOLO 학습 스크립트")

    parser.add_argument("--version", type=int, default=None, help="데이터셋 버전 지정")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolo11n.pt")

    args = parser.parse_args()

    # 2. 데이터셋 경로 결정
    if args.version is not None:
        dataset_path = DATASETS_DIR / f"yolo_doll_v{args.version}"
    else:
        dataset_path = get_latest_dataset_path("yolo_doll")

    if dataset_path is None or not dataset_path.exists():
        print("Error: 학습할 데이터셋을 찾을 수 없습니다.")
        print("먼저 build_dataset.py를 실행했는지 확인하세요.")
        return

    data_yaml = dataset_path / "data.yaml"

    # 3. 장치 선택
    device = select_device()

    print("--- 학습 설정 정보 ---")
    print(f"데이터셋: {dataset_path.name}")
    print(f"사용 장치: {device}")
    print(f"에폭: {args.epochs}, 배치: {args.batch}")
    print("----------------------")

    # 4. 모델 로드 및 학습 시작
    model = YOLO(args.model)

    # 결과는 data/models 아래 데이터셋 이름으로 저장
    save_dir = PROJECT_ROOT / "data" / "models"

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(save_dir),
        name=dataset_path.name,
        exist_ok=True
    )

    print(f"\n학습이 완료되었습니다! 결과 저장소: {save_dir / dataset_path.name}")

if __name__ == "__main__":
    run_training()