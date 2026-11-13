import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import load_labels, imread, MAP, COLORS
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 단계 4 헷갈리는 사례 수집: 결함이 아닌데 결함처럼 보이는 것
# (영상 이름, dacl10k 라벨, 몇 번째 폴리곤, 제목, 혼동되는 결함, 확대할 때 주변 여백 비율)
CASES = [
    ('dacl10k_v2_train_6613', 'JTape', 0, '① 시공 이음부', 'crack', 0.15),
    ('dacl10k_v2_train_0649', 'Drainage', 0, '② 배관', 'crack', 0.15),
    ('dacl10k_v2_validation_0397', 'Graffiti', 3, '③ 점검 표시(분필)', 'crack', 1.2),
    ('dacl10k_v2_train_6563', 'Wetspot', 13, '④ 젖은 얼룩', 'efflorescence', 0.15),
    ('dacl10k_v2_train_2472', 'Weathering', 16, '⑤ 표면 오염·변색', 'efflorescence', 0.6),
    ('dacl10k_v2_train_1352', 'Restformwork', 0, '⑥ 거푸집 잔재', 'spall_detach', 0.15),
]
lab = {x['name']: x for x in load_labels()}

fig, ax = plt.subplots(3, 4, figsize=(14, 10.5))
for k, (n, L, i, title, conf, r) in enumerate(CASES):
    x = lab[n]
    im = imread(f'data/images/{n}.jpg', 1000)
    h, w = im.shape[:2]
    # 실제 결함(3클래스)은 얇은 클래스 색 선, 헷갈리는 요소는 굵은 자홍색 선
    for l, p in x['polys']:
        if l in MAP:
            cv2.polylines(im, [(p * [w, h]).astype(np.int32)], True, COLORS[MAP[l]], 2)
    assert x['polys'][i][0] == L
    pts = (x['polys'][i][1] * [w, h]).astype(np.int32)
    full = im.copy()
    cv2.polylines(full, [pts], True, (255, 0, 255), 5)
    # 확대: 해당 영역 주변을 잘라서 원래 모습 그대로 보여준다
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    m = int(r * max(x1 - x0, y1 - y0)) + 20
    crop = imread(f'data/images/{n}.jpg', 1000)[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
    a, b = ax.ravel()[2 * k], ax.ravel()[2 * k + 1]
    a.imshow(full); a.set_title(f'{title} → {conf}로 오인 가능', fontsize=10)
    b.imshow(crop); b.set_title('확대 (표시 없음)', fontsize=9)
for a in ax.ravel():
    a.axis('off')
plt.figtext(0.5, 0.005, '자홍색: 헷갈리는 요소 / 빨강: crack, 파랑: efflorescence, 노랑: spall_detach (정답 주석)', ha='center', fontsize=9)
plt.tight_layout(rect=(0, 0.02, 1, 1))
plt.savefig('figures/hard_negatives.png', dpi=110)
print('figures/hard_negatives.png 저장')
