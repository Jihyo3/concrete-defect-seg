import pickle
import numpy as np
from ultralytics import YOLO
from utils import load_split, predict_all, onoff, NAMES

# 3주차 단계 2: ON/OFF 평가. val, test 예측을 한 번씩만 해서 logs/preds.pkl에 저장
model = YOLO('best.pt')
data = {}
for split in ['val', 'test']:
    paths, gts = load_split(split)
    preds = predict_all(model, paths)
    data[split] = {'paths': paths, 'gts': gts, 'preds': preds}
    on = [sum(c in g for g in gts) for c in range(3)]
    print(f'{split}: 영상 {len(paths)}장 / 정답 ON 영상 crack {on[0]}, efflorescence {on[1]}, spall_detach {on[2]}')
pickle.dump(data, open('logs/preds.pkl', 'wb'))

# 함수 동작 확인: 검증셋, thr=0.5
print('\n[동작 확인] val, thr=0.5')
v = data['val']
for cid, name in enumerate(NAMES):
    r = onoff(v['preds'], v['gts'], 0.5, cid)
    print(f"{name:14s} TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']} F1={r['f1']:.3f}")

# 예측 confidence 분포 (탐색 구간이 맞는지 확인)
print('\n[val 예측 confidence 분포] 영상별 클래스 최고 confidence')
for cid, name in enumerate(NAMES):
    m = [conf[cls == cid].max() if (cls == cid).any() else 0 for conf, cls in v['preds']]
    print(f'{name:14s} 중앙값 {np.median(m):.3f} / 최대 {np.max(m):.3f}')
