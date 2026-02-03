import cv2
import hashlib
from pathlib import Path

def get_file_hash(file_path):
    return hashlib.md5(file_path.read_bytes()).hexdigest()[:12]

def preprocess_images(input_path, output_path):
    src_dir, dst_dir = Path(input_path).resolve(), Path(output_path).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in src_dir.iterdir() if f.suffix.lower() in valid_exts]

    print(f"대상 경로: {src_dir} | 찾은 이미지 수: {len(image_files)}")

    for f in image_files:
        f_hash = get_file_hash(f)
        new_filename = f"img_{f_hash}{f.suffix}"
        save_path = dst_dir / new_filename

        if save_path.exists():
            continue

        img = cv2.imread(str(f))
        if img is None: continue

        resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(save_path), resized)
        print(f"[전처리] {f.name} -> {new_filename}")

    print("전처리 작업이 완료되었습니다.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    preprocess_images(BASE_DIR / "data" / "raw", BASE_DIR / "data" / "processed")