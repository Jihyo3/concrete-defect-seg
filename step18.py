import json, pickle, shutil
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
from ultralytics import YOLO
from utils import NAMES, COLORS

# 3주차 단계 5: 테스트셋 FP·FN 사례. 클래스별 최대 3장씩 error_cases/에 저장하고 정답·예측 비교 그림을 만든다
best = json.load(open('logs/thresholds.json'))
test = pickle.load(open('logs/preds.pkl', 'rb'))['test']
model = YOLO('best.pt')
SAVE_DIR = Path('error_cases')
shutil.rmtree(SAVE_DIR, ignore_errors=True)
SAVE_DIR.mkdir()

def gt_image(p, cid):
    """정답 폴리곤 중 cid 클래스만 그린다"""
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    over = im.copy()
    lb = Path(p.replace('/images/', '/labels/')).with_suffix('.txt')
    for l in open(lb).read().split('\n'):
        v = l.split()
        if v and int(v[0]) == cid:
            pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.fillPoly(over, [pts], COLORS[NAMES[cid]])
            cv2.polylines(im, [pts], True, COLORS[NAMES[cid]], 3)
    return cv2.addWeighted(over, 0.35, im, 0.65, 0)

def pred_image(p, cid):
    """선택 임계값 이상인 cid 클래스 예측만 그린다"""
    r = model.predict(p, imgsz=960, conf=best[NAMES[cid]], classes=[cid], device='mps', verbose=False)[0]
    return cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), r

picked = []
for cid, name in enumerate(NAMES):
    fp_imgs, fn_imgs = [], []
    for p, gt, (conf, cls) in zip(test['paths'], test['gts'], test['preds']):
        c = conf[cls == cid]
        pred_on = bool((c >= best[name]).any())
        gt_on = cid in gt
        top = float(c.max()) if len(c) else 0.0
        if pred_on and not gt_on:
            fp_imgs.append((top, p))
        if not pred_on and gt_on:
            fn_imgs.append((top, p))
    fp_imgs.sort(reverse=True)   # FP는 가장 확신한 것부터
    fn_imgs.sort()               # FN은 가장 놓친 것부터
    print(f'{name} | FP={len(fp_imgs)} | FN={len(fn_imgs)}')
    class_dir = SAVE_DIR / name
    class_dir.mkdir()
    for error_type, items in [('FP', fp_imgs), ('FN', fn_imgs)]:
        for i, (top, p) in enumerate(items[:3]):
            im, _ = pred_image(p, cid)
            cv2.imwrite(str(class_dir / f'{error_type}_{i+1}_{Path(p).name}'), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            others = [NAMES[g] for g in test['gts'][test['paths'].index(p)]]
            print(f'  {error_type}_{i+1} {Path(p).stem} 최고 conf {top:.3f} / 정답 클래스 {others}')
            if i == 0:
                picked.append((name, error_type, p, top, cid))

# 리포트 그림: 클래스별 대표 FP, FN 한 장씩 (한 줄에 FP 정답·예측, FN 정답·예측)
fig, ax = plt.subplots(3, 4, figsize=(16, 9.6))
for k, (name, et, p, top, cid) in enumerate(picked):
    row, col = k // 2, (k % 2) * 2
    ax[row, col].imshow(gt_image(p, cid))
    ax[row, col + 1].imshow(pred_image(p, cid)[0])
    ax[row, col].set_title(f'{name} {et} 정답\n{Path(p).stem[11:]}', fontsize=10)
    ax[row, col + 1].set_title(f'예측 (thr {best[name]}, 최고 conf {top:.2f})', fontsize=10)
for a in ax.flat:
    a.axis('off')
plt.tight_layout()
plt.savefig('figures/error_cases.png', dpi=110)
