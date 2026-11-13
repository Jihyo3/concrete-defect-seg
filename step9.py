import glob, sys
from pathlib import Path
import numpy as np

# 단계 3 데이터 무결성 점검: 영상-라벨 대응, 빈 파일, 클래스 ID, 좌표 범위, 꼭짓점 수
# python step9.py fix 로 실행하면 고칠 수 있는 항목을 고친 뒤 다시 점검한다
FIX = len(sys.argv) > 1 and sys.argv[1] == 'fix'
G = lambda p: sorted(glob.glob(p, recursive=True))

def check():
    imgs = {Path(p).stem for p in G('dataset/images/**/*.jpg')}
    lbls = {Path(p).stem: p for p in G('dataset/labels/**/*.txt')}
    r = {'라벨 없는 영상': sorted(imgs - set(lbls)), '영상 없는 라벨': sorted(set(lbls) - imgs),
         '빈 라벨 파일(배경 영상)': [], '클래스 ID 오류': [], '좌표 범위 벗어남': [], '꼭짓점 3개 미만': [], '좌표 개수 홀수': []}
    for s, p in lbls.items():
        lines = [ln.split() for ln in open(p) if ln.strip()]
        if not lines:
            r['빈 라벨 파일(배경 영상)'].append(s)
        for v in lines:
            if int(v[0]) not in (0, 1, 2):
                r['클래스 ID 오류'].append(s)
            if (len(v) - 1) % 2:
                r['좌표 개수 홀수'].append(s)
            if len(v) < 7:
                r['꼭짓점 3개 미만'].append(s)
            if any(not 0 <= float(x) <= 1 for x in v[1:]):
                r['좌표 범위 벗어남'].append(s)
    for k, v in r.items():
        print(f'  {k}: {len(set(v))}개' + (f'  예) {sorted(set(v))[0]}' if v else ''))
    return r

print('[점검]')
r = check()
bad = set(r['좌표 범위 벗어남'])
if bad:
    p = f'dataset/labels/{sorted(bad)[0]}.txt'
    v = np.concatenate([[float(x) for x in ln.split()[1:]] for ln in open(p)])
    print(f'  좌표 범위 예시: {Path(p).name} 최소 {v.min():.6f} 최대 {v.max():.6f}')

if FIX:
    # 원본 주석이 영상 테두리를 살짝 넘어간 경우라서 0~1로 자른다
    for s in bad:
        p = f'dataset/labels/{s}.txt'
        out = []
        for ln in open(p):
            v = ln.split()
            out.append(v[0] + ' ' + ' '.join(f'{min(max(float(x), 0), 1):.6f}' for x in v[1:]))
        open(p, 'w').write('\n'.join(out) + '\n')
    print(f'\n[수정] 좌표 범위를 벗어난 {len(bad)}개 파일의 좌표를 0~1로 잘랐다')
    print('\n[다시 점검]')
    check()
