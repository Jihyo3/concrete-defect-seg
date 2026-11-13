import json, pickle
import pandas as pd
from utils import onoff, NAMES

# 3주차 단계 4: 테스트셋 고정 적용. step16에서 저장한 임계값을 그대로 읽어서 쓴다
best = json.load(open('logs/thresholds.json'))
test = pickle.load(open('logs/preds.pkl', 'rb'))['test']

rows = []
for cid, name in enumerate(NAMES):
    r = onoff(test['preds'], test['gts'], best[name], cid)   # 검증셋에서 선택한 임계값 그대로 사용
    rows.append([name, best[name], r['tp'], r['fp'], r['fn'], r['tn'], r['precision'], r['recall'], r['f1']])

df = pd.DataFrame(rows, columns=['class', 'threshold', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'f1'])
print(df.round(3).to_string(index=False))
print('macro F1:', round(df['f1'].mean(), 3))
df.round(3).to_csv('logs/onoff_test.csv', index=False)

