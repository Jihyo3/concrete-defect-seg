# 콘크리트 구조물 결함 세그멘테이션 (Step 3)

콘크리트 표면 사진에서 균열(crack), 백태(efflorescence), 박리·박락(spall_detach)을 YOLOv8s-Seg로 찾아내는 논문을 3주 동안 따라 해 봤다.
1주차에는 데이터를 확인하고 주석 기준서를 만들었고, 2주차에는 라벨을 정리해서 학습했다. 3주차에는 검증셋에서 클래스별 임계값을 정하고 테스트셋으로 평가했다.

- 작성자: 정지효 (게임공학전공, 22114660)
- 리포트: [1주차](report.pdf) · [2주차](report2.pdf) · [3주차 최종](report3.pdf)

## 데이터

과제에서 쓰는 AI Hub 지하구조물 데이터 대신 공개 데이터 [dacl10k](https://huggingface.co/datasets/Voxel51/dacl10k) (CC-BY-4.0)에서 420장을 골라 썼다.
Crack·ACrack은 crack, Efflorescence는 efflorescence, Spalling은 spall_detach로 바꾸고 나머지 라벨은 지웠다.
주석 기준서로 검토해서 판단 보류 69장을 빼고 351장을 남겼고, 원본 ID 단위로 train 280 / val 35 / test 36장으로 나눴다.

## 주차별 코드

| 주차 | 코드 | 하는 일 |
|---|---|---|
| 1주차 | step1.py | 환경 확인, YOLOv8s-Seg 불러오기 |
| | step2.py | dacl10k에서 420장 골라 받기 |
| | step3.py | 데이터 요약, samples9.png |
| | step4.py | 하드 네거티브 사례 그림 |
| | step5.py | 주석 기준 예시 그림, 판단 보류 목록 |
| | step6.py | 1주차 리포트 (report.pdf) |
| 2주차 | step7.py | YOLO 세그멘테이션 라벨로 변환 |
| | step8.py | 라벨 포맷 확인 |
| | step9.py | 무결성 점검 (`fix`를 붙이면 좌표를 0~1로 고침) |
| | step10.py | 원본 ID 단위 분할, data.yaml |
| | step11.py | 학습 (imgsz 960, batch 4, AdamW, 60 에폭, Mosaic·MixUp 끔) |
| | step12.py | 학습 곡선 curve.png |
| | step13.py | 2주차 리포트 (report2.pdf) |
| 3주차 | step14.py | 테스트셋 mask mAP50, mAP50-95 |
| | step15.py | ON/OFF 평가 함수 확인, val·test 추론 저장 |
| | step16.py | 검증셋에서만 클래스별 임계값 탐색, val_threshold_f1.png |
| | step17.py | 고른 임계값을 테스트셋에 그대로 적용 |
| | step18.py | FP·FN 사례 저장 (error_cases/) |
| | step19.py | 최종 리포트 (report3.pdf) |

공통 함수는 utils.py에 있다.

## 결과

| | 결과 | 논문 |
|---|---|---|
| 테스트셋 mask mAP50 | 0.032 | 0.443 |
| 테스트셋 mask mAP50-95 | 0.008 | - |
| 선택 임계값 (crack / efflorescence / spall_detach) | 0.1 / 0.1 / 0.02 | 0.010 / 0.004 / 0.060 |
| 테스트셋 ON/OFF macro F1 | 0.603 | 0.896 |

학습 사진이 280장뿐이라 논문보다 성능이 많이 낮다. 자세한 내용과 오류 분석은 [report3.pdf](report3.pdf)에 정리했다.

## 실행 방법

```bash
pip install -r requirements.txt

# dacl10k 주석 파일 받기
mkdir -p data
curl -L -o data/samples.json https://huggingface.co/datasets/Voxel51/dacl10k/resolve/main/samples.json

# 1주차
python step1.py && python step2.py && python step3.py && python step4.py && python step5.py && python step6.py
# 2주차 (step11 학습은 맥 MPS에서 2시간 20분 정도 걸렸다)
python step7.py && python step8.py && python step9.py fix && python step10.py && python step11.py && python step12.py && python step13.py
cp runs/segment/runs/step3/weights/best.pt best.pt
# 3주차
python step14.py && python step15.py && python step16.py && python step17.py && python step18.py && python step19.py
```

학습한 가중치는 best.pt로 올려 두었다. seed는 42로 고정했고 실행 환경은 macOS (Apple M4, MPS), Python 3.14.7, torch 2.14.0, ultralytics 8.4.210이다.
리포트 PDF는 Chrome headless로 만들어서 step6, step13, step19는 Google Chrome이 깔려 있어야 한다.

*인공지능을 사용하여 작성된 README 입니다.
