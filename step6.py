import os, subprocess, html
import pandas as pd

# 1주차 제출용 PDF: logs/*.txt + 그림 -> report.html -> report.pdf
# 순서: python step1.py ~ step5.py (출력은 logs/stepN.txt) -> python step6.py
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

def code(src):
    return '<pre>' + html.escape(src.rstrip()) + '</pre>'

def out(txt):
    return f'<pre class="out">{txt}</pre>'

summ = pd.read_csv('logs/dataset_summary.csv', index_col=0)
s3 = log(3)
hold = pd.read_csv('hold_list.csv')

body = f'''
<div class="head">
  <div class="kicker">STEP 3 · 1주차 실습 · 제출물</div>
  <h1>결함의 이해와 주석 기준</h1>
  <p class="meta">{AUTHOR}</p>
</div>


<h2>① 환경 확인</h2>
<p>.venv에 ultralytics와 opencv-python을 설치하고 강의 코드(step1.py)로 YOLOv8s-Seg를 불러왔다. 맥북이라 CUDA는 없고 맥 GPU(MPS)는 사용할 수 있다.</p>
{out(log(1))}

<h2>② 데이터 요약</h2>
{table(['항목', '내용'], [
    ['영상 수', '420장 (dacl10k 교량 결함 데이터에서 선택)'],
    ['해상도', '81종류, 긴 변 640~6000픽셀 (중앙값 1511). 많은 순서로 1280×960, 1024×768 각 56장, 1920×1440 46장'],
])}
{table(['데이터 구성', '영상 수', '인스턴스 수'], [[i] + [f'{int(v):,}' for v in summ.loc[i]] for i in ['crack', 'efflorescence', 'spall_detach']] + [['결함 없음 (헷갈리는 요소만 있는 영상)', '60', '-']], cls='half')}
<p>한 영상에 여러 결함이 같이 있는 경우가 있어서 클래스별 영상 수의 합은 420장보다 많다.</p>

<h2>③ 샘플 그림</h2>
{fig('samples9.png', '그림 1. samples9.png (seed 42로 9장 무작위 선택, 제목 아래는 들어 있는 클래스)', 80)}
{out(s3.split(chr(10)+chr(10)+'밝기')[1].join(['밝기', '']) if '밝기' in s3 else '')}
<p>같은 콘크리트 면이어도 사진마다 색과 밝기가 많이 달랐다. 밝기 평균이 59인 어두운 영상부터 195인 밝은 영상까지 있다.
1행 첫 번째 영상처럼 레일, 배관, 보호 설비가 화면 대부분을 차지하는 경우가 있었다.
2행 두 번째, 3행 첫 번째 영상에는 점검자가 균열 옆에 그린 빨간 점선과 노란 분필 선이 있었는데, 균열과 나란히 있어서 처음에는 둘 다 균열로 보였다.</p>

<h2>④ 하드 네거티브</h2>
{fig('figures/hard_negatives.png', '그림 2. 결함처럼 보이지만 결함이 아닌 사례 6종. 왼쪽은 전체 영상, 오른쪽은 표시 없이 확대한 모습', 100)}
{table(['사례', '혼동되는 결함', '설명'], [
    ['① 시공 이음부', 'crack', '벽면을 위아래로 가로지르는 어두운 선이라 균열처럼 보이지만, 자로 그은 듯 곧고 폭이 처음부터 끝까지 일정하다.'],
    ['② 배관', 'crack', '천장을 따라가는 가는 배관이 멀리서 찍히면 꺾이며 이어지는 균열 선과 비슷하게 보인다.'],
    ['③ 점검 표시(분필)', 'crack', '점검자가 균열 옆에 그린 분필·스프레이 선은 균열과 같은 방향으로 이어져 있어 균열로 같이 칠하기 쉽다.'],
    ['④ 젖은 얼룩', 'efflorescence', '물이 스며 나온 자리라 백태와 위치가 비슷하지만, 어둡게 젖어 있을 뿐 흰 침착물은 없다.'],
    ['⑤ 표면 오염·변색', 'efflorescence', '주변보다 밝게 바랜 부분이 흰 침착물처럼 보이지만, 표면이 볼록하게 쌓인 것이 없고 색만 다르다.'],
    ['⑥ 거푸집 잔재', 'spall_detach', '이음 틈에 남은 나무판과 이물질이 콘크리트가 떨어져 나간 자리처럼 울퉁불퉁하게 보인다.'],
])}

<div class="page">
<h2>⑤ 주석 기준서</h2>
{table(['', 'crack (균열)', 'efflorescence (백태)', 'spall_detach (박리·박락)'], [
    ['클래스 정의', '콘크리트 표면이 선 모양으로 갈라진 틈', '물이 스며 나와 표면에 남은 흰색 침착물', '표면층이 들뜨거나 떨어져 나가 속이 드러난 부분'],
    ['포함 기준', '폭이 일정하지 않고 불규칙하게 꺾이며 이어지는 어두운 선', '흰 가루나 굳은 흰 덩어리가 실제로 보이는 곳 (흘러내린 흰 줄 포함)', '골재·철근이 보이거나 단차가 보이는 곳'],
    ['제외 대상', '시공 이음부, 배관·케이블, 점검 표시·낙서, 그림자', '젖은 얼룩, 표면 오염·변색, 보수 재료, 조명 반사', '거푸집 잔재, 앵커 구멍, 보수 모르타르 자리'],
    ['경계 규칙', '선을 따라 얇게 그리고, 짧게 끊긴 곳은 하나로 잇는다', '확실히 흰 부분까지만 그린다', '실제 떨어진 테두리까지만 그리고 들뜸 의심 범위는 뺀다'],
], cls='guide')}
<h3>판단 보류</h3>
<p class="small">아래 영상은 라벨링하지 않고 <code>hold_list.csv</code>에 따로 모아 다시 검토한다.</p>
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
@page { size: A4; margin: 15mm 16mm; }
body { font-family: "Apple SD Gothic Neo", "AppleGothic", sans-serif; font-size: 10pt; line-height: 1.6; color: #1b2433; }
h1 { font-size: 19pt; color: #0b2545; margin: 4px 0 6px; }
h3 { font-size: 10.5pt; color: #1d5c96; margin: 10px 0 4px; break-after: avoid; }
h2 { font-size: 13pt; color: #0b2545; border-bottom: 2px solid #0b2545; padding-bottom: 2px; margin: 18px 0 8px; break-after: avoid; }
.kicker { letter-spacing: 2px; color: #5b7088; font-size: 8.5pt; }
.meta { color: #46586e; font-size: 9pt; margin: 0 0 6px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 8px; font-size: 9pt; break-inside: avoid; }
th { background: #0b2545; color: white; padding: 4px 6px; text-align: left; font-weight: 600; }
td { padding: 3px 6px; border-bottom: 1px solid #dde4ec; vertical-align: top; }
tr:nth-child(even) td { background: #f3f6f9; }
table.half { width: 60%; }
table.guide { font-size: 8pt; line-height: 1.45; }
table.guide td:first-child { font-weight: 600; white-space: nowrap; color: #0b2545; }
figure { margin: 6px 0 10px; text-align: center; break-inside: avoid; }
figcaption { font-size: 8.5pt; color: #5b7088; }
p { margin: 5px 0; } ul { padding-left: 18px; margin: 4px 0; } li { margin: 2px 0; }
.note { font-size: 8.5pt; color: #5b7088; }
.small { font-size: 8.5pt; line-height: 1.45; }
.page { break-before: page; }
.page h2 { margin-top: 0; }
.page figure { margin: 4px 0 0; }
code { font-family: Menlo, monospace; font-size: 8pt; }
pre { background: #f3f6f9; border: 1px solid #dde4ec; border-radius: 4px; padding: 6px 9px; font-size: 8pt; line-height: 1.45;
      white-space: pre-wrap; break-inside: avoid; margin: 6px 0 8px; font-family: Menlo, monospace; }
pre.out { background: #0f1b2b; color: #d6e2f0; border-color: #0f1b2b; }
'''

with open('report.html', 'w') as f:
    f.write(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Step 3 1주차 제출물</title>'
            f'<style>{css}</style></head><body>{body}</body></html>')

chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
subprocess.run([chrome, '--headless', '--disable-gpu', '--no-pdf-header-footer', '--allow-file-access-from-files',
                f'--print-to-pdf={os.path.abspath("report.pdf")}', 'file://' + os.path.abspath('report.html')],
               check=True, capture_output=True)
print('report.pdf 생성')
