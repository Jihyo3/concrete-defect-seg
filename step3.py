import glob, random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from utils import load_labels, MAP, HARD, CLASSES, SEED
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 단계 2~3: 데이터 구성 확인과 영상 관찰 (강의 자료 코드 그대로 + 요약표)
imgs = sorted(glob.glob('data/images/**/*.jpg', recursive=True))
print('영상 수:', len(imgs))
print('예시:', imgs[0] if imgs else '없음')

lab = {x['name']: x for x in load_labels()}

# 해상도 분포
res = Counter(f"{lab[Path(p).stem]['w']}×{lab[Path(p).stem]['h']}" for p in imgs)
print('\n해상도 종류:', len(res))
for r, n in res.most_common(6):
    print(f'  {r}: {n}장')
side = np.array([max(lab[Path(p).stem]['w'], lab[Path(p).stem]['h']) for p in imgs])
print(f'긴 변 픽셀: 최소 {side.min()} / 중앙값 {int(np.median(side))} / 최대 {side.max()}')

# 클래스별 영상 수와 인스턴스(폴리곤) 수
rows = []
for c in CLASSES:
    ins = [sum(MAP.get(l) == c for l, _ in lab[Path(p).stem]['polys']) for p in imgs]
    rows.append([c, sum(i > 0 for i in ins), sum(ins)])
for k, v in HARD.items():
    ins = [sum(l == k for l, _ in lab[Path(p).stem]['polys']) for p in imgs]
    rows.append([f'(헷갈리는 요소) {v}', sum(i > 0 for i in ins), sum(ins)])
summ = pd.DataFrame(rows, columns=['구분', '영상 수', '인스턴스 수']).set_index('구분')
print('\n' + summ.to_string())
none = sum(not any(l in MAP for l, _ in lab[Path(p).stem]['polys']) for p in imgs)
multi = sum(len({MAP[l] for l, _ in lab[Path(p).stem]['polys'] if l in MAP}) >= 2 for p in imgs)
print(f'\n3클래스가 하나도 없는 영상: {none}장 / 2클래스 이상 같이 있는 영상: {multi}장')
summ.to_csv('logs/dataset_summary.csv')

# 샘플 9장 (강의 코드)
random.seed(SEED)
n = min(9, len(imgs))
sample = random.sample(imgs, n)
fig, ax = plt.subplots(3, 3, figsize=(10, 9))
axes = ax.ravel()
for a in axes:
    a.axis('off')
for a, p in zip(axes, sample):
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    a.imshow(im)
    c = sorted({MAP[l] for l, _ in lab[Path(p).stem]['polys'] if l in MAP}) or ['결함 없음']
    a.set_title(Path(p).name[:20] + '\n' + ', '.join(c), fontsize=7)
plt.tight_layout()
plt.savefig('samples9.png', dpi=110)

# 첫 번째 영상의 밝기 통계 + 전체 영상 밝기 분포
gray = cv2.imread(imgs[0], 0)
print('\n밝기 평균:', round(gray.mean(), 1))
print('밝기 표준편차:', round(gray.std(), 1))
b = np.array([[g.mean(), g.std()] for g in (cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_4) for p in imgs)])
print(f'전체 영상 밝기 평균: 최소 {b[:,0].min():.1f} / 중앙값 {np.median(b[:,0]):.1f} / 최대 {b[:,0].max():.1f}')
print(f'밝기 평균 80 미만(어두운 영상): {(b[:,0] < 80).sum()}장')
np.save('logs/brightness.npy', b)
