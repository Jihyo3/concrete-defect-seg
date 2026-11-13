import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import load_labels, imread, draw, MAP
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 단계 5 주석 기준서에 넣을 예시 영상
# 윗줄: 클래스별 예시 / 아랫줄: 경계 규칙과 판단 보류 예시
ROWS = [
    ('dacl10k_v2_train_2161', 'crack: 갈라진 선을 따라 얇게'),
    ('dacl10k_v2_train_1073', 'efflorescence: 흰 침착물이 보이는 곳까지'),
    ('dacl10k_v2_train_3783', 'spall_detach: 떨어져 나간 부분만'),
    ('dacl10k_v2_train_3718', '경계: 들뜸 의심 범위(자홍)는 빼고\n실제 탈락 부분(노랑)만'),
    ('dacl10k_v2_validation_0371', '보류: 망상 균열을 넓은 면으로 칠한 원본 주석\n(선 단위 기준과 달라 재주석 대상)'),
    ('dacl10k_v2_train_5699', '보류: 타격 검사로 찾은 들뜸 범위(자홍)\n영상만으로는 들뜸 여부를 알 수 없음'),
]
lab = {x['name']: x for x in load_labels()}

fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for k, (a, (n, title)) in enumerate(zip(ax.ravel(), ROWS)):
    x = lab[n]
    im = imread(f'data/images/{n}.jpg', 1000)
    h, w = im.shape[:2]
    im = draw(im, x['polys'], labels=list(MAP), thick=2)
    for l, p in x['polys']:
        if l == 'Hollowareas' and k >= 3:   # 아랫줄에만 들뜸 범위 표시
            cv2.polylines(im, [(p * [w, h]).astype(np.int32)], True, (255, 0, 255), 4)
    a.imshow(im); a.set_title(title, fontsize=10); a.axis('off')
plt.tight_layout()
plt.savefig('figures/class_examples.png', dpi=110)
print('figures/class_examples.png 저장')

# 판단 보류 목록: 바로 라벨링하지 않고 다음 주에 다시 볼 영상
import glob
import pandas as pd
b = np.load('logs/brightness.npy')
names = [p.split('/')[-1][:-4] for p in sorted(glob.glob('data/images/**/*.jpg', recursive=True))]
bright = dict(zip(names, b[:, 0]))
hold = []
for n, x in lab.items():
    why = []
    if any(l == 'ACrack' for l, _ in x['polys']):
        why.append('망상 균열이 면으로 칠해져 있음 (선 단위로 재주석 필요)')
    if any(l == 'Hollowareas' for l, _ in x['polys']) and not any(l == 'Spalling' for l, _ in x['polys']):
        why.append('들뜸 범위 표시만 있고 탈락이 보이지 않음')
    if bright[n] < 80:
        why.append(f'어두움 (밝기 평균 {bright[n]:.0f})')
    if why:
        hold.append((n + '.jpg', ' / '.join(why)))
hold = pd.DataFrame(hold, columns=['파일', '보류 이유']).sort_values('파일')
hold.to_csv('hold_list.csv', index=False)
print('판단 보류 영상:', len(hold), '장 -> hold_list.csv')
for k in ['망상', '들뜸', '어두움']:
    print(f'  {k}:', hold['보류 이유'].str.contains(k).sum())
