import torch
from ultralytics import YOLO

# 단계 1 환경 확인: torch 버전, GPU 사용 가능 여부, YOLOv8s-Seg 가중치 로드
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
print('mps (맥 GPU):', torch.backends.mps.is_available())

model = YOLO('yolov8s-seg.pt')   # 처음 실행하면 가중치를 자동으로 받는다
print('model loaded')
print('task:', model.task, '/ 파라미터 수:', f'{sum(p.numel() for p in model.model.parameters()):,}')
