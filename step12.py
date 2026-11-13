import pandas as pd
import matplotlib.pyplot as plt

# 단계 6 학습 곡선 (강의 코드) + 최고 성능 에폭 출력
df = pd.read_csv('runs/segment/runs/step3/results.csv')
df.columns = [c.strip() for c in df.columns]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for c in ['train/seg_loss', 'val/seg_loss']:
    if c in df.columns:
        ax[0].plot(df['epoch'], df[c], label=c)
ax[0].set_title('Segmentation Loss'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss'); ax[0].legend()
for c in ['metrics/mAP50(M)', 'metrics/mAP50-95(M)']:
    if c in df.columns:
        ax[1].plot(df['epoch'], df[c], label=c)
ax[1].set_title('Mask mAP'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('mAP'); ax[1].legend()
plt.tight_layout()
plt.savefig('curve.png', dpi=150)

# best.pt는 ultralytics fitness(0.1×mAP50 + 0.9×mAP50-95, 박스·마스크 합)가 가장 높은 에폭
f = 0.1 * (df['metrics/mAP50(B)'] + df['metrics/mAP50(M)']) + 0.9 * (df['metrics/mAP50-95(B)'] + df['metrics/mAP50-95(M)'])
b = df.loc[f.idxmax()]
print('학습한 에폭 수:', len(df))
print(f"best 에폭: {int(b['epoch'])} / val mask mAP50 {b['metrics/mAP50(M)']:.3f} / mask mAP50-95 {b['metrics/mAP50-95(M)']:.3f}")
print(f"train seg_loss: {df['train/seg_loss'].iloc[0]:.3f} -> {df['train/seg_loss'].iloc[-1]:.3f}")
print(f"val seg_loss: {df['val/seg_loss'].iloc[0]:.3f} -> 최저 {df['val/seg_loss'].min():.3f} (에폭 {int(df.loc[df['val/seg_loss'].idxmin(), 'epoch'])}) -> 마지막 {df['val/seg_loss'].iloc[-1]:.3f}")
