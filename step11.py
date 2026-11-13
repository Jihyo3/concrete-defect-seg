import time
import torch
from ultralytics import YOLO

# 단계 5 학습: 논문 Table 1 설정 그대로 (맥이라 GPU는 mps)
device = 0 if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
model = YOLO('yolov8s-seg.pt')        # 사전학습 가중치
t = time.time()
model.train(
    data='dataset_split/data.yaml',
    epochs=60,
    imgsz=960,          # letterbox 960
    batch=4,
    optimizer='AdamW',
    lr0=0.00015,
    mosaic=0.0,         # Mosaic 끔
    mixup=0.0,          # MixUp 끔
    patience=20,
    seed=42,
    deterministic=True,
    device=device,
    workers=4,
    project='runs',
    name='step3',
    exist_ok=True,
)
print(f'학습 시간: {(time.time() - t) / 60:.1f}분')
