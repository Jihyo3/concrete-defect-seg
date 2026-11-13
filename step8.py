import glob

# 단계 2 포맷 확인 (강의 코드)
lbls = sorted(glob.glob('dataset/labels/**/*.txt', recursive=True))
print('라벨 파일 수:', len(lbls))
lbls = [p for p in lbls if open(p).read().strip()]   # 빈 파일(배경 영상)은 건너뛰고 첫 라벨을 본다
print('첫 라벨 파일:', lbls[0])
line = open(lbls[0], encoding='utf-8').readline().split()
print('토큰 개수:', len(line))
print('앞부분:', line[:9])
if len(line) == 5:
    print('Detection 포맷입니다.')
elif len(line) >= 7 and (len(line) - 1) % 2 == 0:
    print('Segmentation 포맷입니다.')
else:
    print('라벨 형식을 다시 확인하세요.')
