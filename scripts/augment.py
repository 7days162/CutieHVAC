import cv2
import json
import albumentations as A
from pathlib import Path
import shutil

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class DataAugmentor:
    def __init__(self):
        self.transform = A.Compose([
            # 1. 기하학적 변형 (위치/방향)
            A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
            A.Rotate(limit=15, p=0.5),  # 50% 확률로 살짝 회전

            # 2. 화질 변형 (노이즈 & 흐림)
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 9), p=1.0),  # 렌즈 흐림
                A.MotionBlur(blur_limit=(5, 9), p=1.0),  # 카메라 흔들림
            ], p=0.6),  # 전체적으로 60% 확률로 적용

            # 노이즈 추가
            A.GaussNoise(std_range=(0.1, 0.2), p=0.5),

            # 3. 조명 변형
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),

        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def process_all(self, count=2):
        self._clear_previous_augmentations()

        images = sorted(PROCESSED_DIR.glob("*.jpg"))
        print(f"대상 원본 이미지: {len([i for i in images if '_aug_' not in i.name])}장")

        for img_p in images:
            # 이미 변형된 파일은 제외
            if "_aug_" in img_p.name:
                continue

            json_p = img_p.with_suffix(".json")
            if not json_p.exists():
                continue

            print(f"강력한 변형 적용 중...: {img_p.name}")
            self._generate_variants(img_p, json_p, count)

    def _clear_previous_augmentations(self):
        """(옵션) 기존에 생성된 aug 파일들을 삭제하고 깨끗하게 시작"""
        print("기존 증강 데이터(_aug_)를 정리합니다...")
        for f in PROCESSED_DIR.glob("*_aug_*"):
            f.unlink()

    def _generate_variants(self, img_path, json_path, count):
        # 원본 데이터 로드
        image = cv2.imread(str(img_path))
        if image is None: return

        with open(json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # Labelme 좌표 -> Albumentations(COCO) 포맷 변환
        bboxes = []
        for shape in label_data.get('shapes', []):
            pts = shape['points']
            x_min = min(pts[0][0], pts[1][0])
            y_min = min(pts[0][1], pts[1][1])
            width = abs(pts[1][0] - pts[0][0])
            height = abs(pts[1][1] - pts[0][1])
            bboxes.append([x_min, y_min, width, height])

        # 지정된 횟수만큼 새로운 이미지와 라벨 생성
        # 기존과 겹치지 않게 인덱스를 조금 뒤로 미룸
        start_index = 1
        for i in range(start_index, start_index + count):
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=['doll'] * len(bboxes)
            )

            # 원본명_aug_N.jpg
            new_name = f"{img_path.stem}_aug_{i}"

            # 이미지 저장
            cv2.imwrite(str(PROCESSED_DIR / f"{new_name}.jpg"), transformed['image'])

            # JSON 라벨 생성 및 저장
            new_json = label_data.copy()
            new_json['imagePath'] = f"{new_name}.jpg"
            new_json['shapes'] = []

            for idx, bbox in enumerate(transformed['bboxes']):
                shape = label_data['shapes'][idx].copy()

                # 다시 Labelme의 [[x1, y1], [x2, y2]] 형식으로 복원
                shape['points'] = [
                    [bbox[0], bbox[1]],
                    [bbox[0] + bbox[2], bbox[1] + bbox[3]]
                ]
                new_json['shapes'].append(shape)

            with open(PROCESSED_DIR / f"{new_name}.json", "w", encoding="utf-8") as f:
                json.dump(new_json, f, indent=2)


if __name__ == "__main__":
    augmentor = DataAugmentor()

    augmentor.process_all(count=3)
    print("\n노이즈 및 블러 강화 어그멘테이션 끝")