"""
Labelme(json) + 전처리된 이미지(jpg)가 있는 data/processed를
YOLO 학습용 데이터셋(data/datasets/yolo_doll_v1)으로 변환한다.

결과:
data/datasets/yolo_doll_v1/
  images/train, images/val
  labels/train, labels/val
  data.yaml
  meta/split.json  (재현 가능한 train/val 분할 기록)
"""

from __future__ import annotations

import json
import random
import shutil
import hashlib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from typing import Dict, List, Tuple

# 프로젝트 루트 경로(= scripts/의 상위 폴더)를 기준으로 data 경로를 안정적으로 잡는다.
# PyCharm 실행 설정(Working Directory)이 달라도 항상 동일하게 동작하도록 하기 위함.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------
# 설정
# ----------------------------

@dataclass(frozen=True, slots=True)
class BuildConfig:
    # 입력(라벨링 작업 공간)
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # 데이터셋 출력 루트(버전 폴더들이 생성되는 위치)
    datasets_root: Path = PROJECT_ROOT / "data" / "datasets"

    # 데이터셋 기본 이름 (예: yolo_doll_v1, yolo_doll_v2 ...)
    base_name: str = "yolo_doll"

    # train/val 비율
    val_ratio: float = 0.2

    # split 재현성
    seed: int = 42

    # 클래스(현재는 1개 고정)
    class_names: Tuple[str, ...] = ("doll",)

    # 이미지 확장자
    image_ext: str = ".jpg"


def _rect_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """labelme rectangle 2점 -> YOLO 정규화 좌표(xc, yc, w, h)"""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    # 0~1 정규화
    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def _ensure_dirs(out_dir: Path) -> None:
    """출력 폴더 생성"""
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)


def _write_data_yaml(out_dir: Path, class_names: Tuple[str, ...]) -> None:
    """Ultralytics YOLO용 data.yaml 생성"""
    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: {list(class_names)}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def _collect_pairs(processed_dir: Path, image_ext: str) -> List[Tuple[Path, Path]]:
    """processed_dir에서 (jpg, json) 짝을 수집"""
    images = sorted(processed_dir.glob(f"*{image_ext}"))
    pairs: List[Tuple[Path, Path]] = []

    for img in images:
        js = img.with_suffix(".json")
        if js.exists():
            pairs.append((img, js))
        else:
            print(f"[SKIP] json 없음: {img.name}")

    return pairs


def _compute_signature(pairs: List[Tuple[Path, Path]], cfg: BuildConfig) -> str:
    """현재 data/processed 상태(이미지/라벨)에 대한 시그니처를 만든다.

    - 이미지 파일 자체는 용량이 커서 내용을 읽지 않고 메타데이터로만 반영
    - 라벨(json)은 내용이 바뀌면 학습셋이 바뀐 것이므로 내용을 해시 입력에 포함

    이 시그니처가 이전 버전과 동일하면 "동버전 유지"(새 버전 생성 안 함).
    """

    h = hashlib.sha256()

    # 설정값도 시그니처에 포함(설정이 바뀌면 데이터셋이 달라진 것으로 처리)
    h.update(f"val_ratio={cfg.val_ratio};seed={cfg.seed};classes={cfg.class_names};ext={cfg.image_ext}".encode("utf-8"))

    # 파일 목록은 정렬된 상태로 들어와야 재현 가능
    for img_path, json_path in pairs:
        img_stat = img_path.stat()
        json_stat = json_path.stat()

        # 이미지: 파일명 + 크기 + 수정시간(나노초)
        h.update(img_path.name.encode("utf-8"))
        h.update(str(img_stat.st_size).encode("utf-8"))
        h.update(str(getattr(img_stat, "st_mtime_ns", int(img_stat.st_mtime * 1e9))).encode("utf-8"))

        # 라벨(json): 파일명 + 크기 + 수정시간 + 내용
        h.update(json_path.name.encode("utf-8"))
        h.update(str(json_stat.st_size).encode("utf-8"))
        h.update(str(getattr(json_stat, "st_mtime_ns", int(json_stat.st_mtime * 1e9))).encode("utf-8"))
        h.update(json_path.read_bytes())

    return h.hexdigest()


def _extract_version_num(dir_name: str, base_name: str) -> int:
    """디렉터리 이름에서 v번호 추출. 실패하면 -1 반환."""

    prefix = f"{base_name}_v"
    if not dir_name.startswith(prefix):
        return -1
    tail = dir_name[len(prefix):]
    return int(tail) if tail.isdigit() else -1


def _find_latest_dataset_dir(datasets_root: Path, base_name: str) -> Tuple[Path | None, int]:
    """datasets_root 아래에서 가장 최신(가장 큰 v번호) 데이터셋 디렉터리를 찾는다."""

    if not datasets_root.exists():
        return None, 0

    best_dir: Path | None = None
    best_v = 0

    for p in datasets_root.iterdir():
        if not p.is_dir():
            continue
        v = _extract_version_num(p.name, base_name)
        if v > best_v:
            best_v = v
            best_dir = p

    return best_dir, best_v


def _read_manifest_signature(dataset_dir: Path) -> str | None:
    """기존 데이터셋의 manifest.json에서 시그니처를 읽는다."""

    manifest_path = dataset_dir / "meta" / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sig = manifest.get("signature")
        return str(sig) if sig else None
    except Exception:
        return None


def _write_manifest(dataset_dir: Path, cfg: BuildConfig, signature: str, split_map: Dict[str, List[str]]) -> None:
    """manifest.json 기록(데이터셋 재현/추적용)"""

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "signature": signature,
        "processed_dir": cfg.processed_dir.as_posix(),
        "val_ratio": cfg.val_ratio,
        "seed": cfg.seed,
        "class_names": list(cfg.class_names),
        "image_ext": cfg.image_ext,
        "counts": {"train": len(split_map.get("train", [])), "val": len(split_map.get("val", []))},
    }

    (dataset_dir / "meta" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _make_split(pairs: List[Tuple[Path, Path]], val_ratio: float, seed: int) -> Dict[str, List[str]]:
    """재현 가능한 train/val split 생성(파일명 기준)"""
    names = [img.name for img, _ in pairs]
    random.seed(seed)
    random.shuffle(names)

    val_n = max(1, int(len(names) * val_ratio)) if len(names) > 1 else 0
    val_set = set(names[:val_n])
    train = [n for n in names if n not in val_set]
    val = [n for n in names if n in val_set]
    return {"train": train, "val": val}


def _convert_one(
    img_path: Path,
    json_path: Path,
    out_dir: Path,
    split: str,
    label_to_id: Dict[str, int],
) -> None:
    """한 장 변환: 이미지 복사 + yolo 라벨 txt 생성"""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    if not img_w or not img_h:
        print(f"[SKIP] imageWidth/Height 없음: {json_path.name}")
        return

    yolo_lines: List[str] = []
    for shape in data.get("shapes", []):
        label = (shape.get("label") or "").strip()
        if label not in label_to_id:
            # 클래스 목록에 없는 라벨은 무시
            continue

        pts = shape.get("points", [])
        if len(pts) < 2:
            continue

        # rectangle은 보통 2점(좌상/우하)
        (x1, y1), (x2, y2) = pts[0], pts[1]
        xc, yc, bw, bh = _rect_to_yolo(float(x1), float(y1), float(x2), float(y2), int(img_w), int(img_h))

        if bw <= 0 or bh <= 0:
            continue

        cls_id = label_to_id[label]
        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # 이미지 복사
    dst_img = out_dir / "images" / split / img_path.name
    shutil.copy2(img_path, dst_img)

    # 라벨 저장(빈 파일도 생성: YOLO는 빈 txt 허용)
    dst_lbl = out_dir / "labels" / split / f"{img_path.stem}.txt"
    dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")


def main() -> None:
    cfg = BuildConfig()

    if not cfg.processed_dir.exists():
        raise RuntimeError(
            "processed 디렉터리가 없습니다.\n"
            f"- 기대 경로: {cfg.processed_dir}\n"
            "- 해결: data/processed 폴더를 생성하고 전처리 결과(jpg)와 labelme json을 넣어주세요.\n"
            "  (또는 BuildConfig.processed_dir를 실제 경로로 수정)"
        )

    # 입력 (jpg,json) 짝 수집
    pairs = _collect_pairs(cfg.processed_dir, cfg.image_ext)
    if not pairs:
        raise RuntimeError("변환할 (jpg,json) 짝이 없습니다. data/processed 경로/확장자를 확인하세요.")

    # 현재 입력 상태 시그니처 계산
    signature = _compute_signature(pairs, cfg)

    # 최신 버전 확인
    latest_dir, latest_v = _find_latest_dataset_dir(cfg.datasets_root, cfg.base_name)
    latest_sig = _read_manifest_signature(latest_dir) if latest_dir else None

    # ✅ 입력 상태가 이전과 완전히 같으면 동버전 유지
    if latest_dir is not None and latest_sig == signature:
        print("✅ data/processed 변경 없음 → 기존 데이터셋 버전 유지")
        print(f"- 사용 버전: {latest_dir.name}")
        print(f"- 경로: {latest_dir}")
        return

    # 새 버전 생성
    new_v = latest_v + 1 if latest_v > 0 else 1
    out_dir = cfg.datasets_root / f"{cfg.base_name}_v{new_v}"

    _ensure_dirs(out_dir)

    # train/val split 생성 및 기록
    split_map = _make_split(pairs, cfg.val_ratio, cfg.seed)
    (out_dir / "meta" / "split.json").write_text(
        json.dumps(split_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    label_to_id = {name: i for i, name in enumerate(cfg.class_names)}

    # 변환 실행
    name_to_pair = {img.name: (img, js) for img, js in pairs}
    for split in ("train", "val"):
        for img_name in split_map[split]:
            img_path, json_path = name_to_pair[img_name]
            _convert_one(img_path, json_path, out_dir, split, label_to_id)

    # YOLO 학습 설정 파일 생성
    _write_data_yaml(out_dir, cfg.class_names)

    # manifest 기록(시그니처 포함)
    _write_manifest(out_dir, cfg, signature, split_map)

    print("✅ 완료! (새 버전 생성)")
    print(f"- 입력: {cfg.processed_dir}")
    print(f"- 출력: {out_dir}")
    print(f"- data.yaml: {out_dir / 'data.yaml'}")
    print(f"- split 기록: {out_dir / 'meta' / 'split.json'}")
    print(f"- manifest: {out_dir / 'meta' / 'manifest.json'}")


if __name__ == "__main__":
    main()