import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple


class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        # 모델 로드
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> Tuple[int, List]:
        # 라즈베리파이 속도 때문에 320으로 고정
        results = self.model.predict(
            frame,
            imgsz=320,
            verbose=False,
            device='cpu'
        )

        result = results[0]
        boxes = []

        if result.boxes is not None:
            for box in result.boxes:
                # 컨피던스 체크
                conf = float(box.conf.item())
                if conf >= self.conf_threshold:
                    # 좌표 추출 (x1, y1, x2, y2)
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    boxes.append(tuple(coords))

        return len(boxes), boxes