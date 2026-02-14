"""CutieHVAC YOLO 학습 실행 스크립트

이 스크립트는 data/datasets/yolo_doll_vN/ 형태의 데이터셋을 대상으로
Ultralytics YOLO 학습을 실행한다.

사용 예시 (프로젝트 루트에서 실행)
-----------------------------------
python train_yolo.py
python train_yolo.py --version 1
python train_yolo.py --epochs 100 --imgsz 640 --batch 16
python train_yolo.py --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"


def _extract_version_num(dir_name: str, base_name: str) -> int:
    prefix = f"{base_name}_v"
    if not dir_name.startswith(prefix):
        return -1
    tail = dir_name[len(prefix):]
    return int(tail) if tail.isdigit() else -1


def _find_latest_dataset(base_name: str) -> Tuple[Optional[Path], int]:
    if not DATASETS_ROOT.exists():
        return None, 0

    best_dir: Optional[Path] = None
    best_v = 0

    for p in DATASETS_ROOT.iterdir():
        if not p.is_dir():
            continue
        v = _extract_version_num(p.name, base_name)
        if v > best_v:
            best_v = v
            best_dir = p

    return best_dir, best_v


def _resolve_dataset(base_name: str, version: Optional[int]) -> Path:
    # 특정 버전 지정
    if version is not None:
        return DATASETS_ROOT / f"{base_name}_v{version}"

    # 최신 버전 자동 선택
    latest_dir, _ = _find_latest_dataset(base_name)
    if latest_dir is None:
        raise RuntimeError(
            "학습할 데이터셋을 찾지 못했습니다.\n"
            f"- 기대 경로: {DATASETS_ROOT}/{base_name}_vN\n"
            "- 해결: 먼저 python scripts/build_yolo_dataset.py 를 실행하세요."
        )
    return latest_dir


def _normalize_device(device: str) -> str:
    """Ultralytics가 이해할 수 있는 device 문자열로 정리"""

    if device != "auto":
        return device

    # auto면: 가능하면 mps, 아니면 cpu
    selected = "cpu"
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected = "mps"
    except Exception:
        selected = "cpu"

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CutieHVAC YOLO 학습 실행")

    parser.add_argument("--version", type=int, default=None, help="사용할 데이터셋 버전 (예: 1, 2)")
    parser.add_argument("--base-name", type=str, default="yolo_doll", help="데이터셋 기본 이름")

    parser.add_argument("--model", type=str, default="yolo11n.pt", help="베이스 YOLO 모델")
    parser.add_argument("--epochs", type=int, default=50, help="epoch 수")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto / cpu / mps / 0 등 (auto면 사용 가능한 장치로 자동 선택)",
    )
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers")

    parser.add_argument(
        "--project",
        type=str,
        default=str(PROJECT_ROOT / "data" / "models"),
        help="결과 저장 루트 폴더 (기본: data/models)",
    )
    parser.add_argument("--name", type=str, default="train", help="실험 이름")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # device=auto 처리
    args.device = _normalize_device(args.device)

    dataset_dir = _resolve_dataset(args.base_name, args.version)

    # 데이터셋 버전명을 그대로 모델 폴더 이름으로 사용
    dataset_name = dataset_dir.name
    args.name = dataset_name

    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        raise RuntimeError(
            f"data.yaml을 찾지 못했습니다: {data_yaml}\n"
            "build_yolo_dataset.py를 다시 실행하세요."
        )

    print("=== YOLO 학습 시작 ===")
    print(f"데이터셋: {dataset_dir}")
    print(f"모델: {args.model}")
    print(f"epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")
    print(f"device={args.device}, workers={args.workers}")
    print("")

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "ultralytics가 설치되어 있지 않습니다.\n"
            "pip install ultralytics 를 실행하세요.\n"
            f"원인: {e}"
        )

    model = YOLO(args.model)

    # 결과 저장 경로를 절대경로로 보정
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = PROJECT_ROOT / project_path

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        exist_ok=True,
    )

    print("=== 학습 완료 ===")


if __name__ == "__main__":
    main()