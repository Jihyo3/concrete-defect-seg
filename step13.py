import os, subprocess, html
import pandas as pd

# 2주차 제출용 PDF: logs/*.txt + curve.png -> report2.html -> report2.pdf
# 순서: python step7.py -> step8.py -> step9.py fix -> step10.py -> step11.py -> step12.py -> step13.py
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

# 분할 코드: 원본 ID로 묶고 나누는 부분만 잘라서 보여준다
src = open('step10.py').read()
split_src = src[src.index('def get_origin_id'):src.index('shutil.rmtree')]
split_src = split_src[:split_src.index('def dhash')] + split_src[split_src.index('images = sorted'):]
chk_src = src[src.index('# 확인: 원본 그룹이'):src.index('for split in split_ids:\n    lb')]

s7 = log(7).split('\n')
b12 = log(12)

body = f'''
<div class="head">
  <div class="kicker">STEP 3 · 2주차 실습 · 제출물</div>
  <h1>주석과 모델 학습</h1>
  <p class="meta">{AUTHOR}</p>
</div>

<h2>① 주석 결과</h2>
<p>CVAT로 새로 그리지 않고, 이미 폴리곤이 그려져 있는 dacl10k 라벨을 1주차 주석 기준서에 맞춰 검토했다.
판단 보류 영상은 빼고, 3클래스가 아닌 라벨(녹, 공동, 풍화, 낙서 등)은 제외 대상이라 지웠다.
결함이 없는 영상은 배경 영상으로 남겼다.</p>
{out(log(7))}
{table(['구분', '영상 수'], [['검토한 영상', '420'], ['판단 보류로 뺀 영상', '69'], ['데이터셋에 넣은 영상', '351 (결함 있음 296, 배경 55)']], cls='half')}

<h2>② 포맷 확인</h2>
<p>클래스 ID 뒤에 x, y 좌표가 쌍으로 이어져 있어서 Segmentation 포맷이 맞다.</p>
{out(log(8))}

<h2>③ 무결성 점검</h2>
{out(log(9))}
<p>좌표가 0~1을 벗어난 파일이 7개 있었다. 원본 주석이 영상 테두리를 아주 조금(-0.0006 정도) 넘어간 것이라 0~1로 잘라서 고쳤고, 다시 점검했을 때 0개가 되었다.
빈 라벨 파일 55개는 결함이 없는 배경 영상이라 그대로 두었다.</p>

<h2>④ 분할 코드</h2>
<p>노란색 줄이 원본 ID로 묶고, 섞고, 나누는 부분이다. 영상이 아니라 원본 ID 단위로 train, val, test를 정하고, 그 원본에 속한 영상은 모두 같은 split으로 복사한다.
이번 데이터는 증강 영상이 없어서 파일명이 곧 원본 ID다. 그래서 거의 같은 장면을 찍은 사진도 같은 원본으로 묶도록 비슷한 사진 검사를 더했는데, 해당하는 사진은 없었다.</p>
{code(split_src, mark=('def get_origin_id', "return stem.split('_aug')", 'groups.setdefault(get_origin_id', 'parent[root(ids[b])]', 'random.shuffle(origin_ids)', 'train_ids = set(', 'val_ids = set(', 'test_ids = set('))}
{code(chk_src, mark=('겹치는 원본 그룹',))}
{out(log(10))}

<h2 style="break-before:page">⑤ 학습 곡선</h2>
<p>논문 Table 1 설정(사전학습 yolov8s-seg.pt, imgsz 960, batch 4, AdamW, lr0 0.00015, 60 에폭, Mosaic·MixUp 끔)으로 학습했다.</p>
{fig('curve.png', 'curve.png. 왼쪽은 학습·검증 segmentation loss, 오른쪽은 검증셋 mask mAP', 100)}
{out(b12)}
<p>학습 손실은 60 에폭 동안 4.73에서 2.36으로 꾸준히 내려갔다. 검증 손실은 처음 몇 에폭에 크게 떨어진 뒤 20 에폭쯤부터 3.6~3.8에서 거의 그대로였다.
검증 mask mAP50은 30 에폭에 0.109로 가장 높았고 그 뒤로는 0.07~0.10 사이를 오르내렸다. 학습 손실만 계속 줄어서 뒤쪽 에폭은 학습 데이터에 맞춰지는 쪽으로 보인다.
best.pt는 box와 mask 점수를 합친 기준으로 37 에폭 가중치가 저장됐다. 학습 영상이 280장뿐이라 논문 값(test mask mAP50 0.443)보다 많이 낮다. 학습에는 맥 MPS로 약 2시간 18분 걸렸다.</p>

<h2>⑥ 설정 근거</h2>
<p><b>imgsz 960</b> 가는 균열은 폭이 몇 픽셀밖에 안 된다. 데이터셋 영상은 긴 변이 1280픽셀인데 기본값 640으로 줄이면 절반이 되어, 폭 2픽셀짜리 균열은 1픽셀로 줄어 배경과 잘 구분되지 않는다.
논문처럼 960을 써서 가는 균열을 최대한 남기고, 대신 메모리 때문에 batch를 4로 낮췄다.</p>
<p><b>Mosaic, MixUp 끔</b> Mosaic는 영상 4장을 잘라 붙이는데, 붙인 경계가 콘크리트 위의 직선처럼 보여서 이음부나 균열과 헷갈릴 수 있다.
MixUp은 두 영상을 반투명하게 겹치는데, 그러면 흰 백태나 어두운 균열의 색이 흐려져서 결함 경계가 실제로는 없는 모양이 된다.
논문도 같은 이유로 둘 다 껐다.</p>
'''

css = '''
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

with open('report2.html', 'w') as f:
    f.write(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Step 3 2주차 제출물</title>'
            f'<style>{css}</style></head><body>{body}</body></html>')

chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
subprocess.run([chrome, '--headless', '--disable-gpu', '--no-pdf-header-footer', '--allow-file-access-from-files',
                f'--print-to-pdf={os.path.abspath("report2.pdf")}', 'file://' + os.path.abspath('report2.html')],
               check=True, capture_output=True)
print('report2.pdf 생성')
