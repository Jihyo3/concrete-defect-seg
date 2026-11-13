import json, ast, os, random, subprocess
import numpy as np
import cv2

# 대체 공개 데이터: dacl10k (Flotzinger et al., WACV 2024, CC-BY-4.0)
# huggingface.co/datasets/Voxel51/dacl10k · 교량 콘크리트 결함 19종을 폴리곤으로 주석
# 논문 데이터(AI Hub 지하구조물)는 로그인·신청이 필요해서 이 데이터로 대신한다
HF = 'https://huggingface.co/datasets/Voxel51/dacl10k/resolve/main/'
SEED = 42

CLASSES = ['crack', 'efflorescence', 'spall_detach']
# dacl10k 라벨 -> 논문 3클래스. Hollowareas(타격으로 찾은 들뜸 표시)는 영상만으로 판단이 안 돼서 넣지 않는다
MAP = {'Crack': 'crack', 'ACrack': 'crack', 'Efflorescence': 'efflorescence', 'Spalling': 'spall_detach'}
# 결함처럼 보이지만 결함이 아닌 것 (하드 네거티브 후보)
HARD = {'JTape': '이음부 테이프', 'EJoint': '신축 이음부', 'Graffiti': '낙서·표시', 'Drainage': '배수관',
        'Wetspot': '젖은 얼룩', 'Weathering': '풍화·오염', 'Restformwork': '거푸집 잔재', 'PEquipment': '보호 설비'}
COLORS = {'crack': (255, 60, 60), 'efflorescence': (60, 160, 255), 'spall_detach': (255, 190, 0)}

def _p(v):
    return ast.literal_eval(v) if isinstance(v, str) else v

def load_samples(path='data/samples.json'):
    """dacl10k 전체 목록 -> [{name, split, file, w, h, polys:[(라벨, Nx2 정규화 좌표)]}] (주석 있는 것만)"""
    out = []
    for x in json.load(open(path))['samples']:
        g = _p(x.get('semseg_ground_truth_filled'))
        if not g:
            continue
        m = _p(x['metadata'])
        polys = [(q['label'], np.array(r)) for q in g['polylines'] for r in q['points'] if len(r) >= 3]
        out.append(dict(name=x['name'], split=_p(x['tags'])[0], file=x['filepath'], w=m['width'], h=m['height'], polys=polys))
    return out

def download(rel, dst):
    # urllib가 가끔 응답 없이 멈춰서 curl로 시간 제한을 두고 받는다
    if not os.path.exists(dst):
        subprocess.run(['curl', '-sfL', '--retry', '5', '--max-time', '120', '-o', dst + '.part', HF + rel])
        if os.path.exists(dst + '.part'):
            os.rename(dst + '.part', dst)

def load_labels(path='data/labels.json'):
    d = json.load(open(path))
    for s in d:
        s['polys'] = [(l, np.array(p)) for l, p in s['polys']]
    return d

def imread(path, long=None):
    """RGB로 읽고 긴 변을 long 픽셀로 줄인다 (그림용)"""
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if long:
        s = long / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return im

def draw(im, polys, labels=None, fill=0.35, thick=3):
    """정규화 폴리곤을 영상 위에 그린다. 3클래스는 클래스 색, 나머지는 흰 점선 느낌의 흰 선"""
    h, w = im.shape[:2]
    over = im.copy()
    for l, p in polys:
        if labels and l not in labels:
            continue
        pts = (p * [w, h]).astype(np.int32)
        c = COLORS.get(MAP.get(l), (255, 255, 255))
        if l in MAP:
            cv2.fillPoly(over, [pts], c)
        cv2.polylines(im, [pts], True, c, thick)
    return cv2.addWeighted(over, fill, im, 1 - fill, 0)

# ---- 3주차: ON/OFF 평가용 ----
SPLIT_DIR = 'dataset_split'
NAMES = ['crack', 'efflorescence', 'spall_detach']

def load_split(split):
    """split 폴더의 이미지 경로와 영상별 정답 클래스 ID 목록"""
    from pathlib import Path
    paths = sorted(Path(SPLIT_DIR, 'images', split).glob('*.jpg'))
    gts = []
    for p in paths:
        lb = Path(SPLIT_DIR, 'labels', split, p.stem + '.txt')
        lines = open(lb).read().split('\n') if lb.exists() else []
        gts.append(sorted({int(l.split()[0]) for l in lines if l.strip()}))
    return [str(p) for p in paths], gts

def predict_all(model, paths):
    """영상마다 한 번만 추론해 (conf, cls)를 저장한다.
    predict 기본 conf=0.25면 그보다 낮은 임계값을 탐색할 수 없어서 conf=0.001로 낮춘다."""
    preds = []
    for p in paths:
        r = model.predict(p, imgsz=960, conf=0.001, device='mps', verbose=False)[0]
        preds.append((r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)))
    return preds

def onoff(preds, gts, thr, cid_target):
    """강의 onoff 함수. 추론은 predict_all로 미리 해 두고 여기서는 세기만 한다"""
    tp = fp = fn = tn = 0
    for (conf, cid), gt in zip(preds, gts):
        pred_on = bool(((cid == cid_target) & (conf >= thr)).any())
        gt_on = cid_target in gt
        if gt_on and pred_on:
            tp += 1
        elif not gt_on and pred_on:
            fp += 1
        elif gt_on and not pred_on:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall, 'f1': f1}
