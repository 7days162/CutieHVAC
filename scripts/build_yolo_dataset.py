import json
import random
import shutil
import hashlib
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class BuildConfig:
    def __init__(self):
        # 경로 설정
        self.processed_dir = PROJECT_ROOT / "data" / "processed"
        self.datasets_root = PROJECT_ROOT / "data" / "datasets"

        # 기본 설정값
        self.base_name = "yolo_doll"
        self.val_ratio = 0.2
        self.seed = 42
        self.class_names = ["doll"]
        self.image_ext = ".jpg"

def rect_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """labelme 좌표를 YOLO 포맷으로 변환"""
    # 좌표 정렬 (min, max 처리)
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    bw = right - left
    bh = bottom - top
    xc = left + (bw / 2.0)
    yc = top + (bh / 2.0)

    # 0~1 사이로 정규화
    return xc / img_w, yc / img_h, bw / img_w, bh / img_h

def write_data_yaml(out_dir, class_names):
    """YOLO 학습용 설정 파일 생성"""
    path_str = out_dir.as_posix()
    content = [
        f"path: {path_str}",
        "train: images/train",
        "val: images/val",
        f"names: {class_names}"
    ]

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(content))

def compute_data_signature(pairs, cfg):
    """데이터 변경 여부 확인을 위한 해시 계산"""
    sha = hashlib.sha256()

    # 기본 설정값 포함
    setup_info = f"{cfg.val_ratio}-{cfg.seed}-{cfg.class_names}"
    sha.update(setup_info.encode("utf-8"))

    for img_p, json_p in pairs:
        # 파일 메타데이터(수정시간 등) 활용
        img_stat = img_p.stat()
        sha.update(img_p.name.encode("utf-8"))
        sha.update(str(img_stat.st_mtime).encode("utf-8"))

        # JSON 내용은 직접 읽어서 반영
        with open(json_p, "rb") as f:
            sha.update(f.read())

    return sha.hexdigest()

def main():
    cfg = BuildConfig()

    if not cfg.processed_dir.exists():
        print(f"Error: {cfg.processed_dir} 경로를 찾을 수 없습니다.")
        return

    # 이미지/JSON 짝 맞추기
    images = sorted(cfg.processed_dir.glob(f"*{cfg.image_ext}"))
    pairs = []
    for img in images:
        json_file = img.with_suffix(".json")
        if json_file.exists():
            pairs.append((img, json_file))

    if not pairs:
        print("변환할 데이터가 없습니다.")
        return

    # 현재 상태 시그니처 확인
    current_sig = compute_data_signature(pairs, cfg)

    # 버전 관리 로직
    latest_v = 0
    if cfg.datasets_root.exists():
        for d in cfg.datasets_root.iterdir():
            if d.is_dir() and d.name.startswith(cfg.base_name + "_v"):
                try:
                    v_num = int(d.name.split("_v")[-1])
                    if v_num > latest_v:
                        latest_v = v_num
                except:
                    continue

    # 기존 버전과 동일한지 체크 (manifest.json 읽기)
    if latest_v > 0:
        prev_dir = cfg.datasets_root / f"{cfg.base_name}_v{latest_v}"
        manifest_p = prev_dir / "meta" / "manifest.json"
        if manifest_p.exists():
            with open(manifest_p, "r", encoding="utf-8") as f:
                prev_manifest = json.load(f)
                if prev_manifest.get("signature") == current_sig:
                    print(f"데이터 변경 없음: v{latest_v}를 그대로 사용합니다.")
                    return

    # 새 버전 생성 준비
    new_v = latest_v + 1
    out_dir = cfg.datasets_root / f"{cfg.base_name}_v{new_v}"
    print(f"새 데이터셋 버전 생성 중: v{new_v}")

    # 폴더 구조 생성
    for s in ["train", "val"]:
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    # 데이터 분할 (Split)
    random.seed(cfg.seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * (1 - cfg.val_ratio))
    train_data = pairs[:split_idx]
    val_data = pairs[split_idx:]

    # 변환 작업 수행
    label_map = {name: i for i, name in enumerate(cfg.class_names)}

    for split_name, data_list in [("train", train_data), ("val", val_data)]:
        for img_p, json_p in data_list:
            # JSON 읽기
            with open(json_p, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            w = label_data["imageWidth"]
            h = label_data["imageHeight"]

            yolo_labels = []
            for shape in label_data.get("shapes", []):
                label = shape["label"]
                if label in label_map:
                    pts = shape["points"]
                    # Rectangle 좌표 (x1, y1), (x2, y2)
                    y_coords = rect_to_yolo(pts[0][0], pts[0][1], pts[1][0], pts[1][1], w, h)
                    line = f"{label_map[label]} {' '.join([f'{c:.6f}' for c in y_coords])}"
                    yolo_labels.append(line)

            # 파일 복사 및 저장
            shutil.copy2(img_p, out_dir / "images" / split_name / img_p.name)
            txt_name = img_p.stem + ".txt"
            with open(out_dir / "labels" / split_name / txt_name, "w") as f:
                f.write("\n".join(yolo_labels))

    # 마무리 설정 파일들
    write_data_yaml(out_dir, cfg.class_names)

    manifest = {
        "signature": current_sig,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_count": len(train_data),
        "val_count": len(val_data)
    }
    with open(out_dir / "meta" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"완료! 데이터셋 경로: {out_dir}")

if __name__ == "__main__":
    main()