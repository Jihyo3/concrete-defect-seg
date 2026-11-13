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
<p>콘크리트 표면 사진에서 세 가지 결함을 찾는다. crack(균열)은 표면이 선 모양으로 갈라진 틈, efflorescence(백태)는 물이 스며 나와 남은 흰 침착물,
spall_detach(박리·박락)는 표면층이 들뜨거나 떨어져 속이 드러난 부분이다.</p>
<p>세 결함 모두 모양이 제각각이고 균열은 폭이 몇 픽셀밖에 안 돼서 박스로는 위치를 제대로 나타낼 수 없다. 그래서 픽셀 단위로 영역을 그리는 세그멘테이션을 골랐다.</p>
<p>평가는 두 가지로 했다. 마스크가 정답과 얼마나 겹치는지 보는 mask mAP, 그리고 사진에 그 결함이 있는지 없는지만 맞히는 ON/OFF 평가다.
점검 현장에서는 결함이 있는 사진을 먼저 골라내는 것이 중요해서, ON/OFF의 macro F1을 주 목표로 두었다.</p>

<h2>2. 데이터</h2>
<p>과제 데이터(AI Hub) 대신 공개 데이터 dacl10k에서 420장을 받아 썼다. 1주차 주석 기준서로 라벨을 검토해서 판단 보류 69장을 빼고 351장을 남겼다.
Crack·ACrack은 crack, Efflorescence는 efflorescence, Spalling은 spall_detach로 바꾸고 나머지 라벨은 지웠다. 기준서 수정본은 맨 뒤 부록에 있다.</p>
<p>분할은 원본 ID 단위로 8:1:1로 했다. 같은 원본에서 나온 사진이 train과 test에 나뉘어 들어가지 않게 하려는 것이고, 비슷한 사진 검사도 했지만 묶인 쌍은 없었다.</p>
{table(['split', '영상', 'crack 영상', 'efflorescence 영상', 'spall_detach 영상', '배경 영상'], [
    ['train', 280, 118, 125, 178, 42], ['val', 35, 14, 15, 24, 7], ['test', 36, 16, 16, 26, 6]])}

<h2>3. 방법</h2>
<p>사전학습된 YOLOv8s-Seg를 논문 설정(imgsz 960, batch 4, AdamW, lr0 0.00015, 60 에폭, Mosaic·MixUp 끔, seed 42)으로 학습했다.
imgsz를 960으로 한 것은 640으로 줄이면 폭 2픽셀짜리 균열이 1픽셀이 되어 거의 사라지기 때문이다.
Mosaic는 사진을 잘라 붙인 경계가 균열이나 이음부처럼 보이고, MixUp은 백태의 흰색과 균열의 어두운 선을 흐리게 만들어서 껐다.</p>
<p>ON/OFF 평가는 한 사진에서 해당 클래스 예측 중 confidence가 임계값 이상인 것이 하나라도 있으면 예측 ON, 정답 라벨에 그 클래스가 있으면 정답 ON으로 보고 TP·FP·FN·TN을 셌다.
YOLO 기본 conf 0.25로 추론하면 그보다 낮은 임계값을 시험할 수 없어서, conf 0.001로 한 번 추론해 두고 임계값만 바꿔 가며 셌다.</p>
<p>임계값은 검증셋에서만 고르고 테스트셋에는 그대로 썼다. 아래 코드에서 step16은 val만 꺼내 쓰고 고른 값을 파일로 저장하며, step17은 그 파일을 읽어서 테스트셋에 적용만 한다.</p>
{code(s16, mark=("val = data['val']", 'best_idx = next', 'json.dump'))}
{code(s17, mark=('json.load', "best[name], cid"))}

<h2>4. 결과</h2>
<h3>세그멘테이션 성능 (테스트셋)</h3>
{table(['클래스', 'mask mAP50', 'mask mAP50-95'], [
    ['crack', *seg['crack']], ['efflorescence', *seg['efflorescence']], ['spall_detach', *seg['spall_detach']],
    ['전체', seg['all50'], seg['all']]], cls='half')}
<p>테스트셋 mask mAP50은 {float(seg['all50']):.3f}로, 논문(0.443)보다 크게 낮다. 학습 영상이 280장으로 적고, 검증셋에서 나온 0.081보다도 낮아서 사진 수가 적은 영향이 크다고 본다.
세 클래스 모두 비슷하게 낮은데, 균열은 폭이 가늘어서 몇 픽셀만 어긋나도 IoU가 크게 떨어지고, 백태와 박락은 경계가 흐려서 라벨과 예측 범위가 잘 맞지 않았다.</p>

<h3>임계값 F1 곡선 (검증셋)</h3>
{fig('val_threshold_f1.png', '검증셋 클래스별 임계값-F1 곡선, 동그라미가 선택한 임계값', 72)}
{table(['클래스', '선택 임계값', '검증셋 F1', '선택 근거'], [
    ['crack', best['crack'], val_f1['crack'], '0.1에서 F1이 가장 높고, 0.2부터 미탐이 4장에서 7장으로 늘었다'],
    ['efflorescence', best['efflorescence'], val_f1['efflorescence'], '0.1에서 가장 높고, 0.2에서는 미탐이 3장에서 8장으로 늘었다'],
    ['spall_detach', best['spall_detach'], val_f1['spall_detach'], '0.02에서 가장 높다. 검증셋 35장 중 24장에 박락이 있어서 낮은 임계값으로 많이 잡는 쪽이 유리했다']])}
<p>세 클래스의 F1 정점이 서로 달라서 클래스별로 다른 임계값을 썼다. 동점이면 낮은 값을 고르기로 했지만 동점은 없었다.
강의 탐색 구간(0.001~0.3)에 0.4, 0.5를 더했는데, 검증셋 예측 confidence 최고값이 0.8까지 나와서 높은 쪽도 확인하려고 넣었다.</p>

<h3>ON/OFF 결과표 (테스트셋, 임계값 고정)</h3>
{table(['클래스', '임계값', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1'],
       [[r['class'], r['threshold'], r['TP'], r['FP'], r['FN'], r['TN'], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}"] for _, r in onoff_df.iterrows()]
       + [['macro F1', '', '', '', '', '', '', '', f'<b>{macro:.3f}</b>']])}
<p>테스트셋 macro F1은 {macro:.3f}이다(논문 0.896). spall_detach는 Recall이 0.923으로 거의 다 찾았지만, crack과 efflorescence는 결함이 있는 사진의 절반 이상을 놓쳤다.
다만 6. 한계에 적은 것처럼 모든 사진을 ON으로 했을 때보다도 F1이 낮아서, 이번 모델은 사진 선별 능력도 아직 부족하다.</p>

<h2>5. 오류 분석</h2>
{fig('figures/error_cases.png', '클래스별 대표 FP(왼쪽 두 칸)와 FN(오른쪽 두 칸). 각 쌍에서 왼쪽은 정답, 오른쪽은 선택 임계값 이상 예측', 100)}
{table(['클래스', '오탐(FP)', '미탐(FN)'], [
    ['crack<br>FP 4 / FN 10', '벽과 바닥이 만나는 모서리 선을 균열로 잡았다(conf 0.42). 원래 라벨은 이음부 테이프와 풍화뿐이다. 길고 어두운 직선이라 균열과 비슷하게 보였다.',
     '정답 균열이 짧고, 주변에 녹·분필 표시·박락이 같이 있어 묻혔다. 미탐 상위 3장 중 2장은 세 결함이 한 사진에 다 있는 경우였다.'],
    ['efflorescence<br>FP 6 / FN 8', '깨진 모서리 단면이 하얗게 밝아서 백태로 잡았다. 정답에는 박락과 균열만 있다.',
     '가늘게 흘러내린 흰 줄이라 면적이 작고 대비가 낮다. 바로 옆의 넓은 흰 띠(이음부 보수재)와도 색이 비슷해서 모델이 백태로 보지 않았다.'],
    ['spall_detach<br>FP 9 / FN 2', '벽면을 따라 흘러내린 흰 줄(정답은 백태)을 박락으로 잡았다(conf 0.77). 표면이 울퉁불퉁하게 일어난 것처럼 보여서 두 클래스가 섞였다.',
     '박락 범위가 작고, 표면 전체에 기공(작은 구멍)이 많아 떨어진 자리와 구분이 잘 안 됐다.']])}
<p>정리하면 crack 오탐은 이음부·모서리 같은 직선 구조물, efflorescence 오탐은 밝은 단면과 흰 재료, spall_detach 오탐은 백태와의 혼동에서 나왔다.
미탐은 세 클래스 모두 결함이 작고 주변과 대비가 낮을 때 많았다. 이 결과를 보고 주석 기준서의 제외 대상과 경계 규칙을 고쳤다(부록).</p>

<h2>6. 한계</h2>
<ul>
<li>테스트셋이 36장뿐이라 사진 한 장만 바뀌어도 F1이 0.03~0.06씩 움직인다. 결과는 대략적인 경향으로만 봐야 한다.</li>
<li>테스트셋 36장 중 26장에 박락이 있어서, 모든 사진을 ON으로 해도 spall_detach F1이 0.839가 나온다. 이번 결과(0.814)는 이보다 낮아서 박락은 선별 능력이 거의 없다고 봐야 한다. crack과 efflorescence도 모두 ON일 때(0.615)보다 낮았다.</li>
<li>사진에 촬영 거리나 크기 기준이 없어서 균열 폭(mm)이나 박락 면적(cm²) 같은 실제 크기는 잴 수 없다.</li>
<li>모델 하나를 seed 하나로 한 번만 학습했다. 다른 seed나 모델과 비교하지 못했다.</li>
<li>지하구조물 데이터가 아니라 교량 위주의 dacl10k를 썼고, 라벨도 새로 그리지 않고 기존 라벨을 검토만 했다.</li>
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
