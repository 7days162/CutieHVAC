import cv2
import hashlib
from pathlib import Path

# 프로젝트 루트 경로 찾기
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def calculate_hash(file_path):
    """파일 내용으로 고유한 해시값 생성 (중복 방지용)"""
    data = file_path.read_bytes()
    return hashlib.md5(data).hexdigest()[:12]


def run_preprocessing(input_dir, output_dir):
    src_path = Path(input_dir).resolve()
    dst_path = Path(output_dir).resolve()

    # 결과 폴더 없으면 생성
    if not dst_path.exists():
        dst_path.mkdir(parents=True)

    # 처리할 확장자 정의
    valid_formats = ['.jpg', '.jpeg', '.png']

    # 이미지 파일 목록 수집
    files = []
    for f in src_path.iterdir():
        if f.suffix.lower() in valid_formats:
            files.append(f)

    print(f"작업 시작 - 경로: {src_path} (파일: {len(files)}개)")

    for f in files:
        # 파일명 중복을 피하기 위해 해시값 사용
        file_hash = calculate_hash(f)
        new_name = f"img_{file_hash}{f.suffix}"
        save_to = dst_path / new_name

        # 이미 처리된 파일은 스킵
        if save_to.exists():
            continue

        # 이미지 읽기
        img = cv2.imread(str(f))
        if img is None:
            print(f"경고: {f.name}을 읽을 수 없습니다.")
            continue

        resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # 저장
        cv2.imwrite(str(save_to), resized_img)
        print(f"[OK] {f.name} -> {new_name}")

    print("모든 전처리 작업이 완료되었습니다.")


if __name__ == "__main__":
    # 데이터 경로 설정
    raw_data = PROJECT_ROOT / "data" / "raw"
    processed_data = PROJECT_ROOT / "data" / "processed"

    run_preprocessing(raw_data, processed_data)