import random, shutil
from pathlib import Path
import numpy as np
import cv2
import yaml
from utils import CLASSES, SEED

# 단계 4 원본 단위 분할 + data.yaml (강의 코드에 비슷한 영상 묶기를 더했다)
SRC_IMG_DIR = Path('dataset/images')
SRC_LBL_DIR = Path('dataset/labels')
OUT_ROOT = Path('dataset_split')
SPLIT_RATIO = (0.8, 0.1, 0.1)
CLASS_NAMES = dict(enumerate(CLASSES))
random.seed(SEED)

def get_origin_id(stem):
    """증강 영상(abc_aug1.jpg)은 원본(abc)과 같은 ID. 이번 데이터는 증강 영상이 없어서 파일명이 곧 원본 ID"""
    return stem.split('_aug')[0]

def dhash(p):
    g = cv2.resize(cv2.imread(str(p), 0), (17, 16), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).ravel()

images = sorted(SRC_IMG_DIR.glob('*.jpg'))
if not images:
    raise FileNotFoundError('dataset/images 폴더에 jpg 이미지가 없습니다.')

# 원본 ID별 그룹
groups = {}
for img_path in images:
    groups.setdefault(get_origin_id(img_path.stem), []).append(img_path)

# 같은 장소를 거의 같게 찍은 사진도 사실상 같은 원본이라 한 그룹으로 합친다 (dHash 256비트 중 20비트 이하 차이)
# 처음에 64비트로 했더니 어둡고 밋밋한 다른 사진 2쌍이 묶여서 256비트로 늘렸다
ids = list(groups)
h = {i: dhash(groups[i][0]) for i in ids}
parent = {i: i for i in ids}
def root(i):
    while parent[i] != i:
        i = parent[i]
    return i
pairs = 0
for a in range(len(ids)):
    for b in range(a + 1, len(ids)):
        if (h[ids[a]] != h[ids[b]]).sum() <= 20:
            parent[root(ids[b])] = root(ids[a]); pairs += 1
merged = {}
for i in ids:
    merged.setdefault(root(i), []).extend(groups[i])
groups = merged
print(f'영상 {len(images)}장 -> 원본 그룹 {len(groups)}개 (비슷한 사진 {pairs}쌍을 같은 그룹으로 묶음)')

# 원본 그룹 단위로 섞어서 나눈다
origin_ids = sorted(groups)
random.shuffle(origin_ids)
n = len(origin_ids)
n_train = int(n * SPLIT_RATIO[0])
n_val = int(n * SPLIT_RATIO[1])
train_ids = set(origin_ids[:n_train])
val_ids = set(origin_ids[n_train:n_train + n_val])
test_ids = set(origin_ids[n_train + n_val:])
split_ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}

shutil.rmtree(OUT_ROOT, ignore_errors=True)
for split in ['train', 'val', 'test']:
    (OUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)

for split, ids in split_ids.items():
    for origin_id in ids:
        for img_path in groups[origin_id]:
            shutil.copy2(img_path, OUT_ROOT / 'images' / split / img_path.name)
            lbl_path = SRC_LBL_DIR / f'{img_path.stem}.txt'
            if lbl_path.exists():
                shutil.copy2(lbl_path, OUT_ROOT / 'labels' / split / lbl_path.name)
            else:
                print(f'[확인 필요] 라벨 없음: {img_path.name}')

data_yaml = {'path': str(OUT_ROOT.resolve()), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': CLASS_NAMES}
with open(OUT_ROOT / 'data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

# 확인: 원본 그룹이 두 split에 걸치지 않았는지, split별 영상·인스턴스 수
print('원본 그룹 수  train:', len(train_ids), '/ val:', len(val_ids), '/ test:', len(test_ids))
print('겹치는 원본 그룹  train-val:', len(train_ids & val_ids), '/ train-test:', len(train_ids & test_ids), '/ val-test:', len(val_ids & test_ids))
for split in split_ids:
    lb = list((OUT_ROOT / 'labels' / split).glob('*.txt'))
    c = np.bincount([int(ln.split()[0]) for p in lb for ln in open(p) if ln.strip()], minlength=3)
    print(f'{split:5s} 영상 {len(lb):3d}장 / 인스턴스 ' + ', '.join(f'{CLASSES[k]} {c[k]}' for k in range(3)))
print('data.yaml:', OUT_ROOT / 'data.yaml')
