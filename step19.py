import os, subprocess, html
import pandas as pd

# 3주차 최종 리포트: logs/*.txt + 그림 -> report3.html -> report3.pdf
# 순서: step14.py -> step15.py -> step16.py -> step17.py -> step18.py -> step19.py
AUTHOR = '정지효 (게임공학전공, 22114660)'

def log(n):
    p = f'logs/step{n}.txt'
    return html.escape(open(p).read().strip()) if os.path.exists(p) else ''

def table(head, rows, cls=''):
    h = ''.join(f'<th>{c}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'

def fig(path, cap, w=100):
    return f'<figure><img src="{path}" style="width:{w}%"><figcaption>{cap}</figcaption></figure>'

def code(src, mark=()):
    lines = html.escape(src.rstrip()).split('\n')
    return '<pre>' + '\n'.join(f'<mark>{l}</mark>' if any(m in l for m in mark) else l for l in lines) + '</pre>'

def out(txt):
    return f'<pre class="out">{txt}</pre>'

import json

best = json.load(open('logs/thresholds.json'))
onoff_df = pd.read_csv('logs/onoff_test.csv')
macro = onoff_df['f1'].mean()

# 테스트셋 클래스별 mask 성능 (step14 로그에서)
seg = {}
for l in open('logs/step14.txt'):
    v = l.split()
    if v and v[0] in ('crack', 'efflorescence', 'spall_detach'):
        seg[v[0]] = (v[3], v[6])
    elif l.startswith('mask mAP50 '):
        seg['all50'] = l.split(':')[1].strip()
    elif l.startswith('mask mAP50-95'):
        seg['all'] = l.split(':')[1].strip()

# 임계값 탐색·고정 적용 코드: 검증셋만 쓰고, 저장한 임계값을 그대로 읽는 부분
s16 = open('step16.py').read()
s16 = s16[s16.index("data = pickle.load"):s16.index("plt.figure")]
s17 = open('step17.py').read()
s17 = s17[s17.index("best = json.load"):s17.index("df = pd.DataFrame")]

val_f1 = {l.split()[1]: l.split('F1=')[1].rstrip(')\n') for l in open('logs/step16.txt') if l.startswith('>>')}

body = f'''
<div class="head">
  <div class="kicker">STEP 3 · 최종 리포트</div>
  <h1>콘크리트 구조물 결함 세그멘테이션과 영상 단위 선별</h1>
  <p class="meta">{AUTHOR}</p>
</div>

<h2>1. 문제 정의</h2>
<ul>
<li>대상 결함: crack(선 모양으로 갈라진 틈), efflorescence(흰 침착물), spall_detach(표면이 떨어져 속이 드러난 부분)</li>
<li>세그멘테이션을 고른 이유: 결함 모양이 불규칙하고 균열은 폭이 몇 픽셀이라 박스로는 영역을 나타낼 수 없다.</li>
<li>평가 목표: 마스크 정확도(mask mAP)와 결함 사진 선별 능력(ON/OFF macro F1)을 따로 잰다.</li>
</ul>

<h2>2. 데이터</h2>
<p>AI Hub 대신 dacl10k 420장을 1주차 기준서로 검토해 판단 보류 69장을 빼고 351장을 썼다(기준서 수정본은 부록).
원본 ID 단위로 8:1:1 분할해 같은 원본이 train과 test에 섞이지 않게 했다.</p>
{table(['split', '영상', 'crack 영상', 'efflorescence 영상', 'spall_detach 영상', '배경 영상'], [
    ['train', 280, 118, 125, 178, 42], ['val', 35, 14, 15, 24, 7], ['test', 36, 16, 16, 26, 6]])}

<h2>3. 방법</h2>
<ul>
<li>학습: 사전학습 YOLOv8s-Seg, imgsz 960, batch 4, AdamW, lr0 0.00015, 60 에폭, seed 42</li>
<li>imgsz 960: 640으로 줄이면 가는 균열이 1픽셀 수준으로 사라진다.</li>
<li>Mosaic·MixUp 끔: 붙인 경계가 균열처럼 보이고, 겹치면 백태·균열 색이 흐려진다.</li>
<li>ON/OFF: 해당 클래스 예측 중 임계값 이상이 하나라도 있으면 예측 ON, 정답 라벨에 있으면 정답 ON (conf 0.001로 한 번 추론해 두고 임계값만 바꿔 셈)</li>
<li>임계값: 검증셋에서만 고르고(step16) 저장한 값을 테스트셋에 그대로 적용(step17)</li>
</ul>
{code(s16, mark=("val = data['val']", 'best_idx = next', 'json.dump'))}
{code(s17, mark=('json.load', "best[name], cid"))}

<h2>4. 결과</h2>
<h3>세그멘테이션 성능 (테스트셋)</h3>
{table(['클래스', 'mask mAP50', 'mask mAP50-95'], [
    ['crack', *seg['crack']], ['efflorescence', *seg['efflorescence']], ['spall_detach', *seg['spall_detach']],
    ['전체', seg['all50'], seg['all']]], cls='half')}
<p>논문(0.443)보다 크게 낮다. 학습 사진이 280장으로 적고, 가는 균열과 흐린 경계 때문에 IoU가 잘 나오지 않았다.</p>

<h3>임계값 F1 곡선 (검증셋)</h3>
{fig('val_threshold_f1.png', '검증셋 클래스별 임계값-F1 곡선, 동그라미가 선택한 임계값', 72)}
{table(['클래스', '선택 임계값', '검증셋 F1', '선택 근거'], [
    ['crack', best['crack'], val_f1['crack'], 'F1 최고, 0.2부터 미탐이 4장에서 7장으로 늘어남'],
    ['efflorescence', best['efflorescence'], val_f1['efflorescence'], 'F1 최고, 0.2부터 미탐이 3장에서 8장으로 늘어남'],
    ['spall_detach', best['spall_detach'], val_f1['spall_detach'], 'F1 최고, 박락 사진이 많아(24/35) 낮은 값이 유리']])}
<p>클래스마다 F1 정점이 달라서 클래스별 임계값을 썼다. 검증셋 confidence가 0.8까지 나와 강의 구간에 0.4, 0.5를 더해 탐색했다.</p>

<h3>ON/OFF 결과표 (테스트셋, 임계값 고정)</h3>
{table(['클래스', '임계값', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1'],
       [[r['class'], r['threshold'], r['TP'], r['FP'], r['FN'], r['TN'], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}"] for _, r in onoff_df.iterrows()]
       + [['macro F1', '', '', '', '', '', '', '', f'<b>{macro:.3f}</b>']])}
<p>macro F1 {macro:.3f}(논문 0.896). spall_detach는 대부분 찾았지만 crack·efflorescence는 결함 사진의 절반 이상을 놓쳤다.</p>

<h2>5. 오류 분석</h2>
{fig('figures/error_cases.png', '클래스별 대표 FP(왼쪽 두 칸)와 FN(오른쪽 두 칸). 각 쌍에서 왼쪽은 정답, 오른쪽은 선택 임계값 이상 예측', 100)}
{table(['클래스', '오탐(FP)', '미탐(FN)'], [
    ['crack<br>FP 4 / FN 10', '벽과 바닥이 만나는 모서리 선을 균열로 봄', '짧은 균열이 녹·분필 표시·박락 사이에 묻힘'],
    ['efflorescence<br>FP 6 / FN 8', '깨진 모서리의 밝은 단면을 백태로 봄', '가늘고 대비가 낮은 흰 줄, 옆의 흰 보수재와 비슷함'],
    ['spall_detach<br>FP 9 / FN 2', '흘러내린 흰 줄(백태)을 박락으로 봄', '작은 박락이 기공 많은 표면과 구분되지 않음']])}
<p>오탐은 직선 구조물·밝은 재료·다른 결함과의 혼동, 미탐은 작고 대비가 낮은 결함에서 나왔다. 이를 반영해 기준서를 고쳤다(부록).</p>

<h2>6. 한계</h2>
<ul>
<li>테스트셋 36장: 사진 한 장에 F1이 0.03~0.06씩 바뀐다.</li>
<li>모든 사진을 ON으로 해도 F1이 spall_detach 0.839, crack·efflorescence 0.615라 이번 결과보다 높다. 선별 능력이 아직 부족하다.</li>
<li>촬영 거리·크기 기준이 없어 균열 폭(mm) 같은 물리 단위로 잴 수 없다.</li>
<li>모델 하나, seed 하나로 한 번만 학습했다.</li>
<li>지하구조물이 아닌 dacl10k를 썼고, 라벨은 새로 그리지 않고 검토만 했다.</li>
</ul>

<h2>7. 재현 정보</h2>
{table(['항목', '내용'], [
    ['실행 환경', 'macOS, Apple M4 (MPS), Python 3.14.7, torch 2.14.0, ultralytics 8.4.210'],
    ['seed', '42 (데이터 선택, 분할, 학습 모두)'],
    ['data.yaml', 'dataset_split/data.yaml (train/val/test = images/train, images/val, images/test, names 0 crack, 1 efflorescence, 2 spall_detach)'],
    ['모델 설정', 'yolov8s-seg.pt, imgsz 960, batch 4, AdamW, lr0 0.00015, epochs 60, patience 20, mosaic 0, mixup 0'],
    ['가중치', 'best.pt (37 에폭, runs/segment/runs/step3/weights/best.pt 복사본)'],
    ['실행 순서', '1주차 step1~6 → 2주차 step7 → step8 → step9 fix → step10 → step11(학습) → step12 → step13<br>'
                 '3주차 step14(테스트 mAP) → step15(val·test 추론) → step16(val 임계값 탐색) → step17(test 고정 적용) → step18(오류 사례) → step19(리포트)'],
], cls='guide')}

<div class="page">
<h2>부록. 주석 기준서 (수정본)</h2>
<p class="small">1주차 기준서에서 노란색 부분을 3주차 오류 분석 결과를 보고 고쳤다. 고친 기준은 다음 라벨 검토부터 적용한다.</p>
{table(['', 'crack (균열)', 'efflorescence (백태)', 'spall_detach (박리·박락)'], [
    ['클래스 정의', '콘크리트 표면이 선 모양으로 갈라진 틈', '물이 스며 나와 표면에 남은 흰색 침착물', '표면층이 들뜨거나 떨어져 나가 속이 드러난 부분'],
    ['포함 기준', '폭이 일정하지 않고 불규칙하게 꺾이며 이어지는 어두운 선', '흰 가루나 굳은 흰 덩어리가 실제로 보이는 곳 (흘러내린 흰 줄 포함)', '골재·철근이 보이거나 단차가 보이는 곳'],
    ['제외 대상', '시공 이음부, 배관·케이블, 점검 표시·낙서, 그림자, <mark>벽과 바닥이 만나는 모서리·부재 끝선</mark>',
     '젖은 얼룩, 표면 오염·변색, 보수 재료, 조명 반사, <mark>깨진 단면·모서리의 밝은 면, 이음부 보수재 같은 흰 재료</mark>',
     '거푸집 잔재, 앵커 구멍, 보수 모르타르 자리, <mark>흘러내린 흰 줄(백태로 표시)</mark>'],
    ['경계 규칙', '선을 따라 얇게 그리고, 짧게 끊긴 곳은 하나로 잇는다. <mark>다른 결함 위에 있는 균열도 따로 그린다</mark>',
     '확실히 흰 부분까지만 그린다. <mark>박락 단면 안의 흰 부분은 박락으로 본다</mark>',
     '실제 떨어진 테두리까지만 그리고 들뜸 의심 범위는 뺀다. <mark>기공이 많은 면은 단차가 분명한 곳만 그린다</mark>'],
], cls='guide')}
<h3>판단 보류</h3>
<ul class="small">
<li>그물 모양 균열이 면으로 칠해져 있는 영상</li>
<li>들뜸 범위 표시만 있고 떨어진 곳이 보이지 않는 영상</li>
<li>너무 어두워서 균열인지 표면 자국인지 알 수 없는 영상</li>
</ul>
<h3>예시 영상</h3>
{fig('figures/class_examples.png', '윗줄: 클래스별 예시 / 아랫줄: 경계 규칙과 판단 보류 예시', 100)}
</div>
'''

css = '''
table.guide { font-size: 8pt; line-height: 1.45; }
table.guide td:first-child { font-weight: 600; white-space: nowrap; color: #0b2545; }
.page { break-before: page; }
.page h2 { margin-top: 0; }
.small { font-size: 8.5pt; line-height: 1.45; }

@page { size: A4; margin: 15mm 16mm; }
body { font-family: "Apple SD Gothic Neo", "AppleGothic", sans-serif; font-size: 10pt; line-height: 1.6; color: #1b2433; }
h1 { font-size: 19pt; color: #0b2545; margin: 4px 0 6px; }
h2 { font-size: 13pt; color: #0b2545; border-bottom: 2px solid #0b2545; padding-bottom: 2px; margin: 18px 0 8px; break-after: avoid; }
.kicker { letter-spacing: 2px; color: #5b7088; font-size: 8.5pt; }
.meta { color: #46586e; font-size: 9pt; margin: 0 0 6px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 8px; font-size: 9pt; break-inside: avoid; }
table.half { width: 60%; }
th { background: #0b2545; color: white; padding: 4px 6px; text-align: left; font-weight: 600; }
td { padding: 3px 6px; border-bottom: 1px solid #dde4ec; vertical-align: top; }
tr:nth-child(even) td { background: #f3f6f9; }
figure { margin: 6px 0 10px; text-align: center; break-inside: avoid; }
figcaption { font-size: 8.5pt; color: #5b7088; }
p { margin: 5px 0; }
pre { background: #f3f6f9; border: 1px solid #dde4ec; border-radius: 4px; padding: 6px 9px; font-size: 8pt; line-height: 1.45;
      white-space: pre-wrap; break-inside: avoid; margin: 6px 0 8px; font-family: Menlo, monospace; }
pre.out { background: #0f1b2b; color: #d6e2f0; border-color: #0f1b2b; }
mark { background: #ffe58a; }
'''

with open('report3.html', 'w') as f:
    f.write(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Step 3 최종 리포트</title>'
            f'<style>{css}</style></head><body>{body}</body></html>')

chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
subprocess.run([chrome, '--headless', '--disable-gpu', '--no-pdf-header-footer', '--allow-file-access-from-files',
                f'--print-to-pdf={os.path.abspath("report3.pdf")}', 'file://' + os.path.abspath('report3.html')],
               check=True, capture_output=True)
print('report3.pdf 생성')
