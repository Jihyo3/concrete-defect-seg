import json, pickle
import matplotlib.pyplot as plt
from utils import onoff, NAMES

# 3주차 단계 3: 임계값 탐색. 검증셋(val)만 사용한다
data = pickle.load(open('logs/preds.pkl', 'rb'))
val = data['val']   # test는 여기서 열지 않는다

# 강의 구간에 0.4, 0.5를 더함 (val 최고 confidence가 0.8까지 나와서)
THRS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5]

best = {}
all_scores = {}
for cid, name in enumerate(NAMES):
    scores = []
    for t in THRS:
        r = onoff(val['preds'], val['gts'], t, cid)
        scores.append(r['f1'])
        print(f"{name} | thr={t:.3f} | F1={r['f1']:.3f} | FP={r['fp']} | FN={r['fn']}")
    all_scores[name] = scores
    max_f1 = max(scores)
    best_idx = next(i for i, s in enumerate(scores) if s == max_f1)   # 동점이면 낮은 임계값
    best[name] = THRS[best_idx]
    print(f">> {name} 선택 임계값: {THRS[best_idx]} (F1={max_f1:.3f})\n")

print('최종 임계값:', best)
json.dump(best, open('logs/thresholds.json', 'w'))

plt.figure(figsize=(7, 4))
for name in NAMES:
    plt.plot(THRS, all_scores[name], marker='o', label=name)
    plt.scatter([best[name]], [max(all_scores[name])], s=150, facecolors='none', edgecolors='k')
plt.xscale('log')
plt.xlabel('Confidence threshold (log)')
plt.ylabel('F1-score')
plt.title('Validation F1 by Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('val_threshold_f1.png', dpi=150)
