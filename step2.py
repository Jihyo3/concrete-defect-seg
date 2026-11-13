import os, json, random
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from utils import load_samples, download, MAP, HARD, CLASSES, SEED

# 단계 2 데이터 확보: dacl10k에서 실습용으로 일부만 고른다
# - 3클래스(crack / efflorescence / spall_detach)가 들어 있는 영상을 클래스마다 120장씩
# - 3클래스는 없고 헷갈리는 요소(이음부, 낙서, 배수관, 젖은 얼룩 등)만 있는 영상 60장
random.seed(SEED)
s = load_samples()
print('dacl10k 주석 영상 수:', len(s), Counter(x['split'] for x in s))

def cls(x):
    return {MAP[l] for l, _ in x['polys'] if l in MAP}

pick, used = [], set()
for c in CLASSES:
    cand = [x for x in s if c in cls(x) and x['name'] not in used]
    for x in random.sample(cand, 120):
        pick.append(x); used.add(x['name'])
neg = [x for x in s if not cls(x) and {l for l, _ in x['polys']} & set(HARD) and x['name'] not in used]
pick += random.sample(neg, 60)
print('고른 영상 수:', len(pick))

os.makedirs('data/images', exist_ok=True)
with ThreadPoolExecutor(16) as ex:
    list(ex.map(lambda x: download(x['file'], f"data/images/{x['name']}.jpg"), pick))

json.dump([{**x, 'polys': [(l, p.tolist()) for l, p in x['polys']]} for x in pick],
          open('data/labels.json', 'w'))
print('저장: data/images/*.jpg, data/labels.json')
