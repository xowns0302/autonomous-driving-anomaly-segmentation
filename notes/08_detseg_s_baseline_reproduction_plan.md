# DetSeg-S Baseline 재현 계획

- 작성일: 2026-10-12
- 근거: DetSeg 논문(ICCV 2025) Table 2·6 + `external/DetSeg-official` 코드(HEAD `a1aca17b`) 정적 분석
- 상태: 계획 문서 (작성 시점). **실행 결과는 `experiments/result_log.md` Exp-R1 ~ R5, E1 ~ E3와 `notes/09_experiment_record.md` 참고.**
- 관련 문서: [06_detseg_code_analysis.md](06_detseg_code_analysis.md)
- 「확인 필요」: 데이터를 받거나 실행해 봐야 확정할 수 있는 항목

---

## 1. 목적

배포 코드와 checkpoint(`detseg_swin-b_coco_20260623-207453f5.pth`)로 돌린 **DetSeg-S가 논문 Table 2의 IoU를 재현하는지** 확인한다.

이 단계가 개선 실험의 관문이다.

- **재현됨**: box 단계의 실패 지점을 진단한 뒤 개선 실험(IE-NMS, size-aware, road-aware)으로 간다.
- **재현 안 됨**: 개선보다 **논문과 배포 코드의 차이 분석**을 먼저 한다. 특히 attention mask 미적용 가능성(06 문서 발견 #3), IE-NMS bar 단순화, SAM 버전을 본다.

DetSeg-S를 먼저 하는 이유:
- 외부 anomaly score map이 필요 없다.
- 배포 checkpoint에 없는 seg_decoder도 필요 없다.
- 논문 Table 6 ablation도 DetSeg-S IoU로 보고되어 있어서, 나중에 attention mask와 IE-NMS 영향도 같은 지표로 비교할 수 있다.

---

## 2. Dataset 우선순위

| 순위 | Dataset | 장수 | 논문 DetSeg-S IoU | 선정 이유 |
|---|---|---|---|---|
| 1 | **FS L&F** | 100 | 32.36 | 작은 도로 장애물. 개선 주제와 직결. default config가 이미 이 dataset을 가리킨다 |
| 2 | **Road Anomaly** | 60 | 68.35 | 다양한 물체(동물, 돌, 라바콘). 도로 밖 물체도 있어 road-aware 검증에 필요 |
| 3 | **FS Static** | 30 (확인 필요) | 83.34 | 합성 데이터. 논문이 데이터 버전 불일치를 언급해 재현 난이도가 높다 |
| 4 | SMIYC Obstacle val | 30 (확인 필요) | 81.58 | 작은 장애물. test는 라벨 비공개라 서버 제출 필요 |
| 5 | LostAndFound 원본 | 2104 프레임 | (Table 4는 DetSeg-R만) | 하이퍼파라미터 튜닝용 split 후보 |

**실행 순서**: FS L&F 하나로 파이프라인 전체를 먼저 통과시킨다 → Road Anomaly 추가 → Table 2와 비교 → (여유 시) FS Static.

---

## 3. Dataset별 expected directory structure

모든 경로는 DetSeg root(`external/DetSeg-official/`) 기준이다. 현재 `data/` 폴더는 **없다**.

권장 방식: 실제 데이터는 repo 밖(예: `~/datasets/`)에 두고 `data/` 아래에 symlink를 건다.

```bash
mkdir -p external/DetSeg-official/data
ln -s ~/datasets/FS_LostFound  external/DetSeg-official/data/FS_LostFound
```

### 3.1 FS L&F — `data/FS_LostFound/`

```text
data/FS_LostFound/
├── images/        xxx.png   (100장)
└── labels_masks/  xxx.png   (이미지와 같은 파일명)
```

- 사용 클래스: `FSLostAndFoundDataset` (`mmdet/detseg_utils/datasets.py:137`), mmseg `BaseSegDataset` 기반
- 매칭 규칙: `img_suffix='.png'`, `seg_map_suffix='.png'`. **파일명이 같아야** 쌍이 맞는다
- GT 값: 0 = ID, 1 = OoD, 255 = void (metric이 이 규약을 가정함, `metrics.py:680-700`)
- 출처: Mask2Anomaly / RbA 계열 repo가 배포하는 전처리본이 `images/`, `labels_masks/` 구조다 (확인 필요). Fishyscapes 공식 배포본은 형식이 다를 수 있다

### 3.2 Road Anomaly — `data/RoadAnomaly/`

```text
data/RoadAnomaly/
├── frame_list.json                       # 이미지 파일명 리스트
└── frames/
    ├── animals01_xxx.jpg
    ├── animals01_xxx.labels/
    │   └── labels_semantic.png
    └── ...
```

- 사용 클래스: ~~`RoadAnomalyJPGDataset`~~ **(2026-10-13 정정)** EPFL이 WebP판과 JPG판을 둘 다 제공한다. 실제 실험은 WebP판 + `RoadAnomalyDataset`으로 했다 (`experiments/configs/detseg-s_road_anomaly.py`). GT 값은 `{0, 2}`로 확인되어 `UnifyGT({0:0, 2:1})`를 적용했다
  - `RoadAnomalyDataset`(`:75`)은 `.webp` 이미지를 가정한다 (`replace('webp', 'labels')`) → `.jpg` 원본에는 맞지 않는다
- GT 값: 공식 도구들은 **`gt == 1 or gt == 2`를 OoD로 처리**한다 (`tools/eval_box_recall.py:144`, `tools/eval_detseg_r_official.py:325`)
  - 그런데 DetSeg-S의 metric(`BinaryMaskAnomalyIoUMetric`)은 **`gt == 1`만 OoD**로 본다
  - → Road Anomaly 라벨의 OoD 값이 2라면 **`UnifyGT`를 넣지 않으면 IoU가 0에 가깝게 나온다** (가장 흔한 재현 실패 원인 후보)
  - `detseg-s_swin-b.py`에는 `UnifyGT(label_map={0:0, 2:1})`가 **주석 처리**되어 있다 (`:188`)

### 3.3 FS Static — `data/FS_Static/`

```text
data/FS_Static/
├── original/   xxx.(png|jpg)   (확인 필요)
└── labels/     xxx.png
```

- 근거: `tools/eval_detseg_r_official.py`의 FS_Static 항목은 모두 `img_path='original', seg_map_path='labels'`, internal config(`:343`)도 같다
- `detseg-s_swin-b.py`의 FS_Static 줄은 주석이고 prefix는 `images/labels_masks`로 남아 있다 → **반드시 `--cfg-options`로 prefix를 바꿔야 한다**
- 주의: 논문은 "기존 방법들이 쓰던 FS Static val set에 불일치가 있어 일관된 버전으로 다시 평가했다"고 쓴다. **어느 버전인지 코드·문서에 명시가 없다** (확인 필요 → 저자 repo issue)
- 이미지와 라벨 파일명 규칙이 다르면(예: `_rgb.jpg` vs `_labels.png`) BaseSegDataset이 쌍을 못 찾는다 → 로드된 샘플 수를 반드시 확인

### 3.4 (선택) SMIYC — `data/SMIYC/dataset_ObstacleTrack/`

```text
images/        xxx.webp
labels_masks/  xxx_labels_semantic.png
```
- `SMIYCDataset(img_suffix='.webp', split='validation')`. `split='validation'`이면 파일 경로에 `validation`이 들어간 이미지만 사용 (`datasets.py:489-492`)

---

## 4. Config에서 확인한 dataset loading 방식

`configs/detseg/detseg-s_swin-b.py` 기준:

| 항목 | 값 | 비고 |
|---|---|---|
| model | `DetSegS`, `sam_model_name='facebook/sam-vit-base'` | SAM은 HuggingFace에서 첫 실행 시 다운로드 |
| test dataset | `FSLostAndFoundDataset`, `data/FS_LostFound`, prefix `images` / `labels_masks` | `:203-204, 242-249` |
| test pipeline | `LoadImageFromFile(pillow)` → `FixScaleResize(800,1333)` → `LoadAnnotations(bbox, seg)` → `ConcatPrompt` → `PackDetInputs` | `:178-194`. `UnifyGT` 주석 처리됨 |
| text prompt | dataset의 `get_data_info`가 19 class + `custom_entities=True` 설정, `ConcatPrompt`가 `'objects'` 추가 | `datasets.py:203-205` |
| evaluator | `BinaryMaskAnomalyIoUMetric` | IoU, F1 출력 |
| box 최종 필터 | `label==0 & score>0.2` (하드코딩) | `models.py:961`. 논문 τ=0.2와 일치 |
| backbone init_cfg | `Pretrained ckpts/swin_base_...22k.pth` | `Runner.test()`는 `init_weights`를 호출하지 않으므로 **test에는 불필요** (확인함) |
| language model | `bert-base-uncased` | 단일 이미지 실행 때 이미 캐시됨 |

**IoU 계산 방식** (`metrics.py:680-700`): **이미지별 IoU를 구한 뒤 평균**을 낸다 (dataset 전체 픽셀을 합친 IoU가 아님).
- `union == 0`인 이미지(GT도 예측도 없음)는 IoU 0으로 계산되어 평균을 깎는다
- F1 = 2·I / (I + U) = Dice
- 논문 본문은 IoU를 이미지 평균으로 냈는지 전체 기준으로 냈는지 명시하지 않는다 (확인 필요). 같은 repo의 oracle-threshold metric도 같은 방식이라 논문 수치도 이 방식일 가능성이 높다

---

## 5. Evaluation command

### 5.0 실행 전 체크리스트

```bash
cd ~/projects/autonomous-driving-anomaly-segmentation/external/DetSeg-official
conda activate detseg
```

- [ ] `ckpts/detseg_swin-b_coco_20260623-207453f5.pth` 존재
- [ ] `data/<dataset>` symlink 존재
- [ ] **이미지 수 확인**: FS L&F 100, Road Anomaly 60
  ```bash
  ls data/FS_LostFound/images | wc -l
  ls data/FS_LostFound/labels_masks | wc -l
  python -c "import json;print(len(json.load(open('data/RoadAnomaly/frame_list.json'))))"
  ```
- [ ] **GT 라벨 값 확인** (UnifyGT가 필요한지 결정)
  ```bash
  python - <<'EOF'
  import numpy as np, glob
  from PIL import Image
  for pat in ['data/FS_LostFound/labels_masks/*.png',
              'data/RoadAnomaly/frames/*.labels/labels_semantic.png']:
      vals = set()
      for p in sorted(glob.glob(pat))[:20]:
          vals |= set(np.unique(np.array(Image.open(p))).tolist())
      print(pat, sorted(vals))
  EOF
  ```
  - FS L&F 기대값: `{0, 1, 255}`
  - Road Anomaly에 `2`가 있으면 → UnifyGT 필요 (5.2)
- [ ] HuggingFace 접근 가능 (`facebook/sam-vit-base` 첫 다운로드, 약 375MB)
- [ ] GPU 메모리 여유 (Swin-B GDINO + SAM ViT-B, batch 1)

### 5.1 FS L&F (default config 그대로)

```bash
python tools/test.py \
  configs/detseg/detseg-s_swin-b.py \
  ckpts/detseg_swin-b_coco_20260623-207453f5.pth \
  --work-dir work_dirs/repro_detseg_s/fs_lostfound
```

### 5.2 Road Anomaly (override config 필요)

`UnifyGT`를 pipeline에 넣고 dataset 클래스를 바꿔야 해서 `--cfg-options`만으로는 번거롭다. **우리 repo에 override config를 둔다** (external은 gitignore 대상이라 추적이 안 되므로):

`experiments/configs/detseg-s_road_anomaly.py` (예정, 아직 만들지 않음):
```python
_base_ = '../../external/DetSeg-official/configs/detseg/detseg-s_swin-b.py'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='UnifyGT', label_map={0: 0, 2: 1}),   # 5.0의 라벨 확인 결과에 맞춰 조정
    dict(type='ConcatPrompt'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'custom_entities', 'tokens_positive')),
]
test_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='RoadAnomalyDataset',   # (정정) WebP판 사용
        data_root='data/RoadAnomaly',
        pipeline=test_pipeline))
val_dataloader = test_dataloader
```

```bash
python tools/test.py \
  ../../experiments/configs/detseg-s_road_anomaly.py \
  ckpts/detseg_swin-b_coco_20260623-207453f5.pth \
  --work-dir work_dirs/repro_detseg_s/road_anomaly
```

주의:
- `UnifyGT`는 label_map에 없는 값(255 포함)을 **전부 0으로** 바꾼다. Road Anomaly에 void(255)가 있으면 ID로 잘못 계산된다 → 5.0 라벨 확인 필수
- `RoadAnomalyDataset`은 mmengine `BaseDataset`이 아니라 torch `Dataset`이다. `tools/test.py`의 Runner에서 그대로 도는지 확인 필요 (`tools/eval_box_recall.py`는 이 클래스를 직접 쓰므로 동작 사례는 있음)

### 5.3 FS Static

```bash
python tools/test.py \
  configs/detseg/detseg-s_swin-b.py \
  ckpts/detseg_swin-b_coco_20260623-207453f5.pth \
  --work-dir work_dirs/repro_detseg_s/fs_static \
  --cfg-options \
    test_dataloader.dataset.data_root=data/FS_Static \
    test_dataloader.dataset.data_prefix.img_path=original \
    test_dataloader.dataset.data_prefix.seg_map_path=labels
# 이미지가 .jpg면 추가: test_dataloader.dataset.img_suffix=.jpg
```

### 5.4 같이 남길 로그

- `work_dirs/.../*.log`에서 확인할 것:
  - checkpoint 로드 시 **missing / unexpected keys 경고**
  - 로드된 샘플 수
  - IoU, F1
- 실행 환경 기록: `transformers` 4.46.3, `torch` 2.1.0+cu118, `mmsegmentation` 1.2.2, SAM `facebook/sam-vit-base`, GPU RTX 4070 Ti

---

## 6. 논문 Table 2 기준 수치

**DetSeg-S IoU** (threshold search 없음):

| Road Anomaly | FS L&F | FS Static | SMIYC Anomaly | SMIYC Obstacle |
|---|---|---|---|---|
| 68.35 | 32.36 | 83.34 | 75.82 | 81.58 |

비교 대상 S2M(역시 threshold search 없음): 58.39 / 30.46 / 69.99 / 77.54 / 67.64

**Table 6 ablation** (DetSeg-S IoU) — 불일치 원인을 해석할 때 쓰는 기준선:

| UN queries | Attention mask | IE-NMS | Road Anomaly | FS L&F | FS Static |
|---|---|---|---|---|---|
| – | – | – | 61.21 | 19.39 | 38.46 |
| ✓ | – | – | 44.10 | 2.00 | 16.84 |
| ✓ | ✓ | – | 60.89 | 7.29 | 36.60 |
| ✓ | ✓ | ✓ | **68.35** | **32.36** | **83.34** |

해석 팁: 우리 결과가 "mask 없음" 행에 가깝게 나오면 attention mask 미적용 가설을 지지하는 정황이 된다. 다만 우리 설정은 IE-NMS가 켜져 있으므로 정확히 대응하는 행은 Table 6에 없다(UN ✓, mask –, IE-NMS ✓). 그래서 결론은 8.5의 직접 실험으로 내려야 한다.

참고: README는 배포 checkpoint가 **같은 config로 다시 학습한 것**이라 논문 수치와 약간 차이가 날 수 있다고 쓴다. README에는 DetSeg-R 수치만 있고 **DetSeg-S의 checkpoint 결과는 없다**.

---

## 7. Baseline 비교 표 템플릿

**판정 기준 (실행 전에 미리 고정)**:
- |Δ| ≤ 3 → 재현됨
- 3 < |Δ| ≤ 10 → 부분 재현 (checkpoint 재학습 편차 범위일 수 있음)
- |Δ| > 10 → 불일치 → 8절 점검

| Dataset | 기대 장수 | 로드 장수 | 논문 IoU | 우리 IoU | Δ | 우리 F1 | 판정 | 비고 (GT 처리, 경고 등) |
|---|---|---|---|---|---|---|---|---|
| FS L&F | 100 | | 32.36 | | | | | |
| Road Anomaly | 60 | | 68.35 | | | | | UnifyGT: |
| FS Static | 확인 필요 | | 83.34 | | | | | 버전: |

**보조 기록** (개선 단계의 진단에 그대로 재사용):

| Dataset | 이미지당 평균 OoD box 수 | OoD box 0개인 이미지 수 | IoU=0인 이미지 수 | 이미지당 추론 시간 |
|---|---|---|---|---|
| FS L&F | | | | |
| Road Anomaly | | | | |

이 보조 지표는 `tools/test.py` 기본 출력에 없다. 실행 후 별도 스크립트로 뽑는다(다음 단계에서 작성).

---

## 8. Baseline 불일치 시 점검 항목

**비용이 낮은 것부터** 확인한다. 앞 단계에서 원인이 나오면 뒤 단계는 생략해도 된다.

### 8.1 데이터 / GT 처리 (가장 먼저)
- [ ] 로드된 샘플 수 = 기대 장수인가 (suffix·파일명 불일치로 조용히 줄어들 수 있음)
- [ ] GT 값 규약: OoD = 1, void = 255인가. Road Anomaly는 UnifyGT를 적용했는가
- [ ] UnifyGT가 void를 0으로 바꿔 버리지 않았는가
- [ ] FS Static이 논문이 말한 "일관된 버전"인가 (확인 필요)

### 8.2 지표 정의
- [ ] 이미지별 평균 IoU인가, 전체 픽셀 IoU인가 → 둘 다 계산해서 어느 쪽이 논문에 가까운지 본다
- [ ] `union == 0` 이미지가 있는가 (있으면 평균이 내려감)

### 8.3 Checkpoint mismatch
- [ ] 로드 로그에 `universal_*`, `bbox_head.objectness_branches.*`가 missing으로 나오지 않는가 (key 존재는 06 문서에서 확인함)
- [ ] 배포 ckpt는 재학습본이라 논문 수치와 다를 수 있다 (README 명시). DetSeg-R README 표에서 FS L&F 편차가 최대 약 6 FPR95 포인트 → S에서도 비슷한 크기의 편차는 가능하다

### 8.4 전처리 / prompt
- [ ] 입력 resize `FixScaleResize(800,1333)`: 논문은 "Grounding DINO와 동일"이라고만 쓴다
- [ ] text = `('objects', 19 class)`, `custom_entities=True`
- [ ] 최종 threshold 0.2 = 논문 τ

### 8.5 Attention mask 미적용 가능성 (06 문서 발견 #3)
- 근거: `query_text_mask`(`models.py:1129`)가 decoder의 text cross-attention(`grounding_dino_layers.py:111-116`)으로 전달되지 않는다
- 논문 Table 6: mask가 빠지면 FS L&F 32.36 → 크게 하락
- 확인 방법 (코드 수정이 필요하므로 **baseline 측정 이후** 별도 실험으로 진행):
  1. 정적 확인: decoder layer에 forward hook을 걸어 UN query의 text attention weight가 3번 이후 token에서 0인지 본다
  2. 기능 확인: mask를 실제로 적용하는 패치 버전을 만들어 IoU를 비교한다. 오르면 배포 코드 결함 가능성, 같거나 내려가면 ckpt가 이미 mask 없는 상태로 학습되었을 가능성 (학습도 같은 코드로 했다면 학습·추론이 일관되게 "mask 없음"이다)

### 8.6 IE-NMS bar 차이
- 논문: Cityscapes 일부로 class별 bar를 보정한다 (예: car 0.36, truck 0.58, bicycle 0.28)
- 코드: `[0.2]*11 + [0.5]*8` 고정, λ=2 (`models.py:522, 528`)
- 확인 방법: bar를 스윕(예: thing class 0.3/0.4/0.5/0.6)해 IoU 민감도를 본다. 크게 흔들리면 불일치 원인 후보가 된다

### 8.7 SAM / transformers
- [ ] **논문은 SAM 모델 크기(ViT-B/L/H)를 명시하지 않는다.** 코드는 `facebook/sam-vit-base`다. 논문이 ViT-H였다면 IoU 차이가 클 수 있다 → `sam_model_name=facebook/sam-vit-huge`로 1회 비교 (확인 필요)
  ```bash
  --cfg-options model.sam_model_name=facebook/sam-vit-huge
  ```
- [ ] SAM 3개 multimask 중 첫 번째만 사용 (`models.py:328`, `multimask_output=True` 기본값 확인함). 논문 구현도 그랬는지는 알 수 없음
- [ ] `transformers` 버전: requirements는 `>=4.38,<5`, 우리 환경은 4.46.3

### 8.8 환경
- [ ] 결정론성: 같은 명령을 2번 돌려 IoU가 같은지 (nondeterministic op 여부)

---

## 9. 다음 단계

1. **데이터 확보**: FS L&F → Road Anomaly 순서. 출처와 파일 구조를 3절과 대조
2. **5.0 체크리스트**: 특히 GT 라벨 값
3. **FS L&F 실행 (5.1)** → 7절 표 작성
4. **Road Anomaly**: override config 작성 후 실행 (5.2)
5. **판정**
   - 재현됨 → 보조 지표 + `tools/eval_box_recall.py --min-area`로 면적별 box recall → **단계별 recall 손실 진단**(encoder top-k / IE-NMS / 최종 threshold) → 진단 결과에 맞는 개선안 선택
   - 불일치 → 8절 순서대로 점검. 8.5(attention mask)와 8.7(SAM 크기) 결과는 그 자체로 보고서의 재현성 분석 장이 된다
6. 결과를 `experiments/result_log.md`에 기록하고, 판정에 따라 07 방향 문서를 갱신

### 이번 단계에서 하지 않는 것
- 코드 수정 (attention mask 패치, IE-NMS 인자화 등은 baseline 측정 후)
- DetSeg-R 평가 (외부 score map 또는 seg branch 학습이 필요)
- VPHM (배포 코드에 동작하는 구현 없음)
