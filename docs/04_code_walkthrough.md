# 04. 코드 흐름 설명

이 문서는 "이미지 한 장이 들어가서 결과가 나오기까지 어떤 코드가 무엇을 하는가"를 설명한다.
줄 번호는 `external/DetSeg-official` HEAD `a1aca17b` 기준.

---

## 1. 누가 만든 코드인가

| 코드 | 만든 사람 | 위치 | 우리가 고쳤나 |
|---|---|---|---|
| DetSeg 공식 코드 | 논문 저자 | `external/DetSeg-official/` | **안 고침** |
| └ DetSeg 핵심 | | `mmdet/detseg_utils/models.py` (거의 전부 여기) | |
| └ 데이터 읽기 | | `mmdet/detseg_utils/datasets.py` | |
| └ 채점 | | `mmdet/detseg_utils/metrics.py` | |
| └ 실행 진입점 | | `tools/test.py` | |
| └ 나머지 `mmdet/` | OpenMMLab (MMDetection 3.3) | Grounding DINO 원본 구현 | |
| Mask2Anomaly 공식 코드 | M2A 저자 | `external/Mask2Anomaly/` | **안 고침** |
| 우리 설정 파일 | 우리 | `configs/` | 데이터 경로만 바꿈 |
| 우리 스크립트 | 우리 | `src/` | 측정·진단용 |

원칙: 공식 코드는 그대로 두고, 바꿔야 할 때는 **우리 스크립트 안에서 함수 하나만 바꿔 끼운다** (아래 5절).

---

## 2. 실행하면 무슨 일이 일어나나

```
python tools/test.py configs/detseg-s_fs_lostfound.py ckpts/detseg_swin-b_coco_....pth
```

1. **config 읽기**: `configs/detseg-s_*.py` → 공식 `detseg-s_swin-b.py`를 상속. 모델 종류(`DetSegS`), 데이터 위치, 전처리 순서, 채점 방식이 적혀 있다
2. **모델 만들기 + 가중치 불러오기**: config의 `type='DetSegS'`를 보고 `models.py`의 `DetSegS` 클래스를 만들고 checkpoint를 넣는다
3. **이미지마다 반복**: 데이터 읽기 → 전처리 → `DetSegS.predict()` → 결과 저장
4. **채점**: 모든 이미지가 끝나면 `metrics.py`의 채점 클래스가 IoU 등을 계산해 출력

---

## 3. 이미지 한 장의 여행 (DetSeg-S)

### ① 데이터 읽기·전처리 — `datasets.py` + config의 `test_pipeline`

| 단계 | 하는 일 | 코드 |
|---|---|---|
| 파일 목록 | `images/0.png` ↔ `labels_masks/0.png` 짝 찾기 | `FSLostAndFoundDataset` (`datasets.py:137`) |
| 텍스트 지정 | "찾을 물체 이름" = Cityscapes 19개 class (road, car, person …) | 같은 클래스 `get_data_info` (`:203`) |
| 이미지 읽기·축소 | 2048×1024 → 약 1333×666 | `LoadImageFromFile`, `FixScaleResize` |
| 정답 읽기 | 정답 mask (0 = 정상, 1 = 이상, 255 = 무시) | `LoadAnnotations` |
| (RA만) 정답 값 변환 | 이상 = 2 → 1 | `UnifyGT` |
| 텍스트 앞에 'objects' 붙이기 | ('objects', road, …, bicycle) | `ConcatPrompt` (`datasets.py:24`) |

### ② 물체 찾기 — `models.py`의 `DetSeg` (Grounding DINO 확장)

```
이미지 ──► Swin-B (이미지 특징)  ┐
                                  ├─► encoder (이미지·글자 섞기) ─► 후보 위치 고르기 ─► decoder 6층 ─► head
텍스트 ──► BERT (글자 특징)      ┘
```

| 단계 | 쉬운 설명 | 코드 |
|---|---|---|
| 특징 추출 | 이미지와 글자를 숫자 벡터로 바꾼다 | `extract_feat`, `language_model` |
| 'objects' 글자 강화 | 'objects' 글자에 학습된 universal embedding을 더한다 | `forward_transformer` (`:1026`) |
| 후보 위치 고르기 | **ID용 900개**(글자와 잘 맞는 위치), **UN용 300개**(물체 같아 보이는 위치)를 따로 고른다 | `pre_decoder` (`:1068`, `:1081`) |
| 두 팀 분리 | ID query와 UN query가 서로를 못 보게 막는다 (self-attention mask) | `pre_decoder` (`:1124`) |
| box·점수 계산 | ID query → 19개 class 점수, UN query → "물체다" 점수(objectness) | `GroundingDINOHeadWithUniversalObjectness.forward` (`:415`) |

### ③ 후보 정리 — `predict_by_feat` (`:482`)

- ID 쪽 상위 300개: label = 1~19 (class), 일부는 label 0 ('objects' 글자에 가장 맞는 경우)
- UN 쪽 상위 300개: label = 0
- 합쳐서 최대 600개 → IE-NMS로

### ④ IE-NMS — `_ie_nms` (`:518-541`) ★ 핵심

```python
bar = [0.2]*11 + [0.5]*8          # stuff(road~sky)는 0.2, thing(person~bicycle)은 0.5  ← 하드코딩
확실한_ID = (label != 0) and (score > bar)
확실한_ID의 점수 ×2               # "이건 확실히 아는 물체"
겹침(IoU>0.5)이 있으면 점수 높은 쪽만 남기는 NMS (class 구분 없이)
점수 원래대로 되돌림
```

→ 강화된 ID box가 같은 자리의 UN box를 지운다. 강화 못 받은 ID box는 UN box와 일반 경쟁.

### ⑤ 최종 걸러내기 — `DetSeg.predict` (`:961`)

```python
keep = (score > 0.2) and (label == 0)    # τ = 0.2, ID box는 모두 버림
```
→ 남은 box = **이상 물체 후보 (OoD box)**

### ⑥ mask 만들기 — `DetSegS.predict` (`:1749`) + `SamPromptMixin` (`:256`)

- OoD box를 하나씩 SAM(ViT-B)에 "이 상자 안의 물체를 따라 그려줘"로 넣는다
- 나온 mask들을 합친다(OR) → 최종 이상 물체 mask

### ⑦ 채점 — `BinaryMaskAnomalyIoUMetric` (`metrics.py:637`)

- 이미지마다 IoU = 겹친 면적 / 합친 면적 (무시 픽셀 제외)
- 전체 이미지의 IoU를 평균 → 출력 (논문 Table 2의 IoU)

---

## 4. DetSeg-R은 어디가 다른가 — `DetSegR.predict` (`:1410`)

①~⑤는 거의 같다. 다른 점:

| 단계 | DetSeg-R |
|---|---|
| 입력 추가 | `GetAnomalyScoreMap`(`datasets.py:521`)이 기존 방법(M2A)의 점수 지도 `.npy`를 같이 읽는다 |
| ⑤ | `box_score_thr`(기본 0.2)로 걸러냄 (config에서 바꿀 수 있음) |
| ⑥ 대신 | OoD box **안쪽 픽셀 점수에 +1.0** (`_boost_anomaly_score`, `:1363`) → box 밖 오탐이 상대적으로 낮아짐 |
| ⑦ 대신 | `AnomalyMetricLoad`(AUROC·AP·FPR95), `OracleThresholdAnomalyIoUMetric`(threshold를 0.01씩 바꿔 가며 최고 IoU) |

우리 E2(M2A 단독)는 같은 config에서 `box_score_thr=1.1`로 두어 box가 하나도 안 남게 해서, 점수 지도를 그대로 채점한 것이다.

---

## 5. 우리 스크립트 (`src/`)

### 5.1 바꿔 끼우기 방식

공식 파일은 안 고치고, 스크립트가 실행될 때 **함수 하나를 같은 이름의 다른 함수로 교체**한다(monkeypatch).

```python
GroundingDINOHeadWithUniversalObjectness._ie_nms = 우리_버전   # 공식 파일은 그대로
```

- 교체한 함수에 **원래 값(0.5)을 넣으면 공식 결과(14.26)와 똑같이 나오는 것**을 확인했다 → 교체가 원본과 같은 동작을 한다는 증거

### 5.2 파일별 역할

| 파일 | 역할 | 쓰인 실험 | 공식 코드와의 관계 |
|---|---|---|---|
| `m2a_save_scores.py` | M2A로 이미지별 점수 지도(.npy) 생성. 계산식은 M2A 공식 `anomaly_inference.py`와 같음 | E2, E3 | M2A 공식 코드를 불러서 사용 (`m2a` 환경) |
| `eval_detseg_s_diagnostics.py` | DetSeg-S를 돌리면서 이미지별 IoU, box, 물체 단위 지표를 저장. 옵션으로 bar 등 바꿔 끼우기 | E4 ③④, (개선 단계) | `_ie_nms` 교체 (옵션 줄 때만) |
| `gdino_objects_baseline.py` | 원본 Grounding DINO에 'objects'만 넣고 SAM으로 mask (Table 6 첫 행) | E4 ① | 공식 클래스 재사용 |
| `diag_ie_nms_sources.py` | IE-NMS 직전 후보 600개를 이미지마다 저장 + 최종 box 출처 추적 | A2 | `_ie_nms`를 같은 로직 + 기록으로 교체 |
| `analyze_ie_nms_effect.py` | 저장한 후보로 "IE-NMS가 무엇을 지우고 남겼나" 분류 (GPU 불필요) | A2 | 오프라인 |
| `diag_small_object_loss.py` | 정답 물체가 어느 단계에서 사라졌나 판정 | A2 | 오프라인 |
| `sweep_ie_nms_bars.py` | IE-NMS 로직을 그대로 옮긴 함수 + bar 바꿔 다시 계산하는 도구 | A2, (개선 단계) | 오프라인 |
| `calibrate_ie_nms_bars.py` | 논문 방식 bar 보정 (Cityscapes) | A3 (준비됨) | `_ie_nms` 입력 기록 |

### 5.3 설정 파일 (`configs/`)

| 파일 | 내용 |
|---|---|
| `detseg-s_{dataset}.py` | 공식 DetSeg-S config 상속 + 데이터 경로·확장자만 변경. Road Anomaly는 정답 값 변환(UnifyGT) 추가 |
| `_detseg-r_m2a_base.py` | 공식 DetSeg-R external config 상속 + 채점 2개(픽셀 지표, 최적 threshold IoU) |
| `detseg-r_m2a_{dataset}.py` | 위 base + 데이터 경로 + M2A 점수 지도 폴더 |

---

## 6. 실험과 코드 연결 한눈에

```
E1  tools/test.py + configs/detseg-s_*.py                        → DetSeg-S IoU
E2  src/m2a_save_scores.py → tools/test.py + detseg-r_m2a_* (box 0개) → M2A 단독
E3  같은 점수 지도 → tools/test.py + detseg-r_m2a_*             → M2A + DetSeg-R
E4  gdino_objects_baseline.py / eval_detseg_s_diagnostics.py      → ablation
A2  diag_ie_nms_sources.py → analyze_ie_nms_effect.py, diag_small_object_loss.py
A3  calibrate_ie_nms_bars.py dump → calibrate                      → 논문 방식 bar
```
