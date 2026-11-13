from pathlib import Path
from ultralytics import YOLO

# 3주차 단계 1: 테스트셋 세그멘테이션 성능 (강의 코드, 경로만 이 폴더에 맞춤)
BASE_DIR = Path.cwd()
DATA_YAML = BASE_DIR / "dataset_split" / "data.yaml"
BEST_PT = BASE_DIR / "best.pt"   # runs/segment/runs/step3/weights/best.pt 를 복사한 것

if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {DATA_YAML}")
if not BEST_PT.exists():
    raise FileNotFoundError(f"best.pt를 찾을 수 없습니다: {BEST_PT}")

model = YOLO(str(BEST_PT))
results = model.val(data=str(DATA_YAML), split="test", imgsz=960, batch=4, plots=True,
                    device="mps", project="runs", name="test", exist_ok=True)

class_names = ["crack", "efflorescence", "spall_detach"]
for i, name in enumerate(class_names):
    print(f"{name:14s} mask mAP50: {results.seg.ap50[i]:.4f}  mask mAP50-95: {results.seg.maps[i]:.4f}")

print("mask mAP50   :", round(results.seg.map50, 4))
print("mask mAP50-95:", round(results.seg.map, 4))
print("box mAP50    :", round(results.box.map50, 4))
