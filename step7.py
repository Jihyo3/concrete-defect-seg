import os, shutil
from collections import Counter
import pandas as pd
import cv2
from utils import load_labels, MAP, CLASSES

# 단계 1 주석: 기존 dacl10k 폴리곤을 1주차 주석 기준서대로 검토해서 YOLO segmentation 라벨로 바꾼다
# - 판단 보류 영상(hold_list.csv)은 넣지 않는다
# - 3클래스가 아닌 라벨(이음부, 배관, 낙서 등 제외 대상)은 지운다. 이 영상은 배경 영상으로 남긴다
# - 학습 속도를 위해 영상 긴 변을 1280으로 줄여 저장한다 (좌표는 0~1이라 그대로 쓸 수 있다)
LONG = 1280
ID = {c: i for i, c in enumerate(CLASSES)}   # 0 crack, 1 efflorescence, 2 spall_detach

hold = set(pd.read_csv('hold_list.csv')['파일'].str[:-4])
lab = load_labels()
shutil.rmtree('dataset', ignore_errors=True)
os.makedirs('dataset/images'); os.makedirs('dataset/labels')

keep, drop = Counter(), Counter()
for x in lab:
    if x['name'] in hold:
        continue
    im = cv2.imread(f"data/images/{x['name']}.jpg")
    s = LONG / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"dataset/images/{x['name']}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for l, p in x['polys']:
        if l not in MAP:
            drop[l] += 1
            continue
        keep[MAP[l]] += 1
        lines.append(f'{ID[MAP[l]]} ' + ' '.join(f'{v:.6f}' for v in p.ravel()))
    open(f"dataset/labels/{x['name']}.txt", 'w').write('\n'.join(lines) + ('\n' if lines else ''))

n = len(os.listdir('dataset/images'))
print(f'검토한 영상: {len(lab)}장 / 판단 보류로 뺀 영상: {len(hold)}장 / 데이터셋에 넣은 영상: {n}장')
print('클래스별 인스턴스 수:')
for c in CLASSES:
    print(f'  {ID[c]} {c}: {keep[c]}')
print('지운 제외 대상 폴리곤:', sum(drop.values()), '개', dict(drop.most_common(6)))
