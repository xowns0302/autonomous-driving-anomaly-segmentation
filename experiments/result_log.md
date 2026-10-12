# Result Log

| Date | Exp | Method | Dataset | AUROC | AP | FPR95 | Notes |
|---|---|---|---|---|---|---|---|

## DetSeg 단일 이미지 Inference 실험 — Phase 8~9

### 날짜

2026-10-12

### 목적

DetSeg official code가 로컬 WSL2 + RTX 4070 Ti 환경에서 정상적으로 실행되는지 확인하고, 단일 도로 이미지에 대해 bbox 및 anomaly score map을 생성할 수 있는지 확인한다.

### 실행 환경

- OS: WSL2 Ubuntu 22.04
- Conda environment: detseg
- Python: 3.8.20
- GPU: NVIDIA GeForce RTX 4070 Ti
- PyTorch: 2.1.0+cu118
- MMCV: 2.1.0
- MMEngine: 0.10.7
- mmdet: 3.3.0

### 사용 코드

- Official repo: external/DetSeg-official
- Config: configs/detseg/detseg-r_swin-b_internal.py
- Checkpoint: ckpts/detseg_swin-b_coco_20260623-207453f5.pth

### 입력 이미지

- demo_inputs/road.png

### 단일 이미지 inference 수정 사항

기본 `inference_detector` 실행 시 `KeyError: 'text'`가 발생했다.

원인:

- DetSeg는 Grounding DINO 계열 구조를 사용하므로 text prompt가 필요하다.
- 단일 이미지 inference에서는 `text` key가 자동으로 생성되지 않는다.
- `ConcatPrompt` transform이 `results['text']`를 요구하면서 에러가 발생했다.

해결:

- `ConcatPrompt` 앞에 단일 이미지용 `SetDetSegPrompt` transform을 임시 삽입했다.
- Cityscapes 19개 class prompt를 입력했다.

### 결과

생성된 결과 파일:

- assets/qualitative_results/road_boxes.png
- assets/qualitative_results/road_anomaly_heatmap.png
- assets/qualitative_results/road_overlay.png

정량 로그:

- num boxes total: 13
- num boxes kept: 13
- anomaly score min: -18.806081771850586
- anomaly score max: -13.129426956176758

### 결과 해석

이번 실험은 full benchmark evaluation이 아니다.

즉 다음을 의미하지 않는다.

- Fishyscapes / RoadAnomaly 정량 평가 완료
- AUROC / AP / FPR95 재현 완료
- 논문 성능 재현 완료

이번 실험의 의미는 다음과 같다.

- DetSeg official code가 로컬 GPU 환경에서 실행 가능함을 확인했다.
- 단일 이미지 입력에 대해 bbox prediction을 생성할 수 있음을 확인했다.
- anomaly score map을 생성하고 heatmap/overlay 이미지로 저장할 수 있음을 확인했다.

### 다음 작업

- 결과 이미지 확인
- threshold 조정 실험
- inference script를 `src/` 또는 `experiments/` 아래에 정리
- dataset 기반 evaluation 가능 여부 검토


---

## DetSeg-S baseline 재현 — FS L&F (Exp-R1)

### 날짜

2026-10-12

### 목적

배포 코드와 checkpoint로 DetSeg-S를 돌려 논문 Table 2의 FS L&F IoU(32.36)가 재현되는지 확인한다. 계획: `notes/08_detseg_s_baseline_reproduction_plan.md`

### 설정 (원본 그대로, 코드 수정 없음)

- Config: `configs/detseg/detseg-s_swin-b.py` (default)
- Checkpoint: `ckpts/detseg_swin-b_coco_20260623-207453f5.pth`
- SAM: `facebook/sam-vit-base` (config 기본값)
- Data: Mask2Anomaly 배포 `Validation_Dataset/FS_LostFound_full` → `data/FS_LostFound` symlink
  - 100장, 2048×1024, GT 값 `{0, 1, 255}`, OoD 없는 이미지 1장
- 명령: `python tools/test.py configs/detseg/detseg-s_swin-b.py ckpts/detseg_swin-b_coco_20260623-207453f5.pth --work-dir work_dirs/repro_detseg_s/fs_lostfound`
- 환경: torch 2.1.0+cu118, transformers 4.46.3, mmsegmentation 1.2.2, RTX 4070 Ti
- 로그: `external/DetSeg-official/work_dirs/repro_detseg_s/fs_lostfound/`

### 결과

| Dataset | 기대 장수 | 로드 장수 | 논문 IoU | 우리 IoU | Δ | 우리 F1 | 판정 |
|---|---|---|---|---|---|---|---|
| FS L&F | 100 | 100 | 32.36 | **14.26** | **−18.10** | 18.53 | **불일치** (\|Δ\| > 10) |

- 실행은 정상 종료(exit 0). checkpoint missing/unexpected key 경고 없음
- 추론 시간 약 1초/장 (SAM 포함)

### 해석 (아직 원인 미확정)

- 논문 수치의 약 44% 수준. 판정 기준상 **불일치** → 08 문서 8절 순서로 원인을 점검한다
- 참고로 논문 Table 6에서 attention mask 없음 + IE-NMS 없음은 2.00, mask 있음 + IE-NMS 없음은 7.29이다. 14.26은 어느 행과도 바로 대응되지 않으므로, 이 숫자만으로 attention mask가 원인이라고 말할 수 없다

### 다음 작업

1. 저비용 점검: 지표 정의(전체 픽셀 IoU도 계산), 이미지별 IoU 분포, 이미지당 OoD box 수
2. Road Anomaly 실행 (UnifyGT override 필요)
3. 8절 후순위 점검: SAM 크기(`sam-vit-huge`), attention mask, IE-NMS bar

---

## DetSeg-S baseline 진단 — FS L&F + Road Anomaly (Exp-R2)

### 날짜

2026-10-12

### 설정

- 원본 코드·checkpoint 그대로. 진단 스크립트 `src/eval_detseg_s_diagnostics.py`(DetSeg 코드 수정 없음)로 이미지별 IoU, box, GT component를 저장
- Road Anomaly: EPFL 공식 WebP판, `experiments/configs/detseg-s_road_anomaly.py` (RoadAnomalyDataset + `UnifyGT({0:0, 2:1})`, GT 값 `{0, 2}` 확인)
- 출력: `external/DetSeg-official/work_dirs/diag_detseg_s/{fs_lostfound,road_anomaly}/` (`summary.json`, `per_image.json`, `pred_masks/`)

### 결과

| Dataset | 논문 IoU | 우리 IoU (이미지 평균, 공식 지표) | Δ | 판정 | 전체 픽셀 IoU | F1 |
|---|---|---|---|---|---|---|
| FS L&F | 32.36 | **14.26** | −18.10 | **불일치** | 6.19 | 18.53 |
| Road Anomaly | 68.35 | **76.46** | +8.11 | **부분 재현** (논문보다 높음) | 76.85 | 84.07 |

- 진단 스크립트의 FS L&F IoU가 `tools/test.py` 결과(14.26)와 정확히 같다 → 스크립트 검증됨

| 진단 지표 | FS L&F | Road Anomaly |
|---|---|---|
| 이미지당 OoD box 수 (평균) | 10.5 | 10.2 |
| GT와 겹치지 않는 box 비율 (IoU<0.1) | **90.5%** (951/1051) | 62.4% (382/612) |
| 예측 mask 면적 / GT 면적 (이미지 중앙값) | **47.4배** | 1.06배 |
| IoU = 0인 이미지 | **44 / 100** | 0 / 60 |
| GT component 중앙 면적 (px) | 664 | 427* |
| box recall (IoU≥0.5), 면적 <32² | 32.8% | 23.2%* |
| box recall, 32²~96² | 78.6% | 86.0% |
| box recall, ≥96² | 75.0% | 78.9% |

\* Road Anomaly GT에는 1px 수준 잡음 component가 많아 작은 구간 수치는 신뢰도가 낮다.

### 관찰

1. **FS L&F의 실패 원인은 "ID 물체 위의 FP box"다.** 시각화(`assets/qualitative_results/baseline_detseg_s/fs_lostfound_fp_examples.jpg`)에서 FP box 대부분이 승용차, 트럭, 밴, 자전거 같은 **Cityscapes ID class 물체** 위에 있다. 자차 보닛 위 box도 있다. IE-NMS가 제거해야 할 box가 남아 있는 것이다
2. FP box의 score 분포(중앙값 0.34)가 TP(0.33)와 거의 같다 → **최종 threshold 조정만으로는 분리되지 않는다**
3. FS L&F는 GT가 매우 작아서(중앙 664px) FP box 몇 개가 union을 지배해 IoU가 무너진다. Road Anomaly는 물체가 커서 같은 수준의 FP가 IoU에 거의 영향을 주지 않는다 → **두 dataset의 상반된 결과가 같은 원인으로 설명된다**
4. 작은 GT(<32²)의 box recall이 30% 안팎으로 낮다 → 작은 물체 누락도 별도 문제로 존재

### 해석 (가설)

ID 물체가 OoD로 남는 경로는 두 가지 가능성이 있다 (아직 구분 안 됨).
- (a) UN box는 차를 잡았는데, ID branch의 car box가 bar(thing class 0.5)를 넘지 못해 IE-NMS가 강화하지 못함
- (b) ID query가 universal token('objects')을 볼 수 있어서(attention mask 미적용), ID branch 자체가 차를 label 0('objects')으로 분류해 OoD로 넘김 → 논문 Table 6의 "mask 없음" 상황과 같은 계열

### 다음 작업

- 남은 각 OoD box가 **어느 branch(ID의 'objects' label vs UN)에서 왔는지**, 겹치는 ID box의 class와 score가 얼마였는지 기록 → (a)/(b) 구분
- 결과에 따라 attention mask 적용 실험 또는 IE-NMS bar 실험으로 진행

---

## IE-NMS 출처 진단 — FS L&F (Exp-R3)

### 날짜

2026-10-12

### 방법

- `src/diag_ie_nms_sources.py`: `_ie_nms`를 **동일 로직 + 입력/keep 기록**으로 monkeypatch (DetSeg 코드 수정 없음). detection만 실행(SAM 없음)
- 최종 OoD box(label 0, score>0.2)마다 출처 branch, GT 겹침, IE-NMS 이전 ID-class box 중 가장 많이 겹치는 것의 class/score/bar 통과 여부를 기록
- 검증: 최종 box 수 1051 = Exp-R2와 동일
- 출력: `external/DetSeg-official/work_dirs/diag_ie_nms/fs_lostfound/final_boxes.json`

### 결과

| | 전체 1051 | FP-like 951 (GT IoU<0.1) | TP-like 100 |
|---|---|---|---|
| 출처: UN branch | 926 | **876 (92%)** | 50 |
| 출처: ID branch 'objects' label | 125 | 75 (8%) | 50 |
| ID-class box와 IoU≥0.5로 겹침 | 910 (86.6%) | **860 (90.4%)** | 50 (50%) |
| 겹친 ID box가 bar 통과(강화됨) | **0** | **0** | 0 |
| 겹친 ID box score [p10, p50, p90] | 0.12 / 0.25 / 0.45 | 0.13 / **0.26** / 0.45 | 0.10 / 0.12 / 0.39 |
| 겹친 ID box class (상위) | | car 439, person 86, truck 69, pole 56, bicycle 55, bus 47 | pole 12, motorcycle 10, traffic sign 9 |

### 결론

- **가설 (a) 지지**: FP의 92%는 UN box이고, 그중 90%는 **같은 위치에서 ID branch가 class를 맞게 잡은 box(car, truck, person…)가 있다.** 그런데 그 ID box의 score(중앙값 0.26)가 코드의 bar(thing class 0.5)를 **하나도 넘지 못해** IE-NMS 강화가 한 번도 일어나지 않았다 → UN box가 NMS에서 이긴다
- **가설 (b)는 부차적**: ID branch가 차를 'objects'로 분류해 남은 경우는 FP의 8%
- 즉 FS L&F 재현 실패의 직접 원인은 **IE-NMS bar가 이 checkpoint의 ID score 분포보다 너무 높게 하드코딩된 것**이다. 논문은 bar를 Cityscapes로 보정한 "매칭 쌍의 최소 score"라고 설명하는데(예: car 0.36), 코드의 0.5는 그와 다르다
- 주의: TP(진짜 장애물)와 겹친 ID box도 50% 있다(pole, motorcycle, traffic sign, score 중앙값 0.12). bar를 낮추면 FP는 줄지만 일부 TP도 억제될 수 있다 → **class별 bar의 trade-off**가 핵심

### 다음 작업

- IE-NMS bar 스윕(thing class 0.5 → 0.4/0.3/0.2, class별 보정) → FP 수, box recall, DetSeg-S IoU 측정

---

## IE-NMS bar 탐색 실험 (Exp-E1, 탐색용 — 최종 보고용 아님)

### 날짜

2026-10-12

### 주의

**탐색 실험이다.** bar 값을 평가셋(FS L&F, Road Anomaly)을 보면서 골랐으므로, 이 수치를 "개선 결과"로 보고하면 안 된다. 최종 bar는 별도 보정 데이터(이상 물체가 없는 주행 이미지)로 정해야 한다.

### 방법

- box 수준: `src/sweep_ie_nms_bars.py` — IE-NMS 이전 후보를 저장해 두고 bar만 바꿔 오프라인으로 재계산 (SAM 없음)
- mask IoU: `src/eval_detseg_s_diagnostics.py --thing-bar/--stuff-bar/--never-boost` — `_ie_nms`를 같은 로직 + 설정 가능한 bar로 교체
  - 검증: `--thing-bar 0.5`(원본 값)로 FS L&F IoU 14.26 재현
- "obstacle-like never": pole, traffic light, traffic sign, fence는 강화하지 않음 (진짜 장애물과 겹친 ID box가 이 class들이었기 때문)

### Box 수준 (GT component ≥ 10px, recall = box IoU ≥ 0.5)

FS L&F:

| config | box/img | FP/img | recall | <32² | 32²~96² | ≥96² |
|---|---|---|---|---|---|---|
| 원본 (stuff .2 / thing .5) | 10.51 | 9.51 | 50.0 | 32.8 | 78.6 | 75.0 |
| thing .4 | 8.59 | 7.64 | 47.3 | 31.9 | 76.8 | 56.2 |
| thing .3 | 6.53 | 5.59 | 46.8 | 31.9 | 75.0 | 56.2 |
| thing .2 | 4.61 | 3.69 | 45.7 | 30.2 | 75.0 | 56.2 |
| thing .1 | 3.23 | 2.34 | 44.1 | 28.4 | 73.2 | 56.2 |
| thing .2 + obstacle-like never | 4.86 | 3.91 | 47.3 | 31.0 | 78.6 | 56.2 |
| IE-NMS 강화 없음 | 15.80 | 14.77 | 51.6 | 33.6 | 82.1 | 75.0 |

Road Anomaly:

| config | box/img | FP/img | recall | <32² | 32²~96² | ≥96² |
|---|---|---|---|---|---|---|
| 원본 | 10.20 | 6.37 | 58.3 | 34.5 | 86.0 | 78.9 |
| thing .3 | 8.88 | 5.25 | 55.0 | 29.4 | 84.0 | 77.5 |
| thing .2 | 7.92 | 4.38 | 53.8 | 29.4 | 80.0 | 76.1 |
| thing .2 + obstacle-like never | 7.98 | 4.45 | 53.8 | 29.4 | 80.0 | 76.1 |

### Mask IoU (DetSeg-S, SAM 포함)

| config | FS L&F IoU (이미지 평균) | FS L&F 전체 픽셀 IoU | FS L&F IoU=0 이미지 | RA IoU | RA 전체 픽셀 IoU |
|---|---|---|---|---|---|
| 원본 (thing .5) | 14.26 | 6.19 | 44 | 76.46 | 76.85 |
| **thing .2** | **20.96 (+6.70)** | **26.63 (×4.3)** | 52 | **78.68 (+2.22)** | 80.65 |
| thing .2 + obstacle-like never | 20.53 | 24.48 | 51 | 77.55 | 80.10 |
| (논문 Table 2) | 32.36 | – | – | 68.35 | – |

### 관찰

1. bar를 낮추면 FP가 크게 줄고(FS L&F 9.51 → 3.69/img), **두 dataset 모두 mask IoU가 오른다.** 진단(Exp-R3)에서 찾은 원인이 맞았다는 근거
2. 대가로 box recall이 4~5포인트 떨어진다. FS L&F 큰 물체 recall 75 → 56은 GT 16개 중 3개 차이라 표본이 작다
3. **"obstacle-like never" 보호는 효과가 없었다** (IoU 소폭 하락). 이 class들은 stuff bar(0.2)라 원래도 거의 강화되지 않아서 바꿀 여지가 적다. 아이디어로는 채택하지 않는다
4. FS L&F는 개선 후에도 **IoU=0 이미지가 52장**이고 중앙값이 0이다. FP는 줄었지만 **작은 장애물 자체를 못 잡는 문제**(<32² recall 약 30%)가 남은 병목이다
5. 논문 FS L&F 32.36과의 차이는 18.1 → 11.4로 줄었다. 남은 차이는 작은 물체 recall 쪽으로 보인다 (확인 필요)

### 다음 작업

- 보정 데이터 확보 → 논문 방식(매칭 쌍의 최소 score)으로 class별 bar를 **평가셋과 분리해** 산출
- 작은 물체 recall이 어느 단계에서 손실되는지 진단 (encoder top-300 / IE-NMS / 최종 threshold 0.2)

---

## 작은 물체 손실 단계 진단 + size-aware threshold (Exp-R4, Exp-E2 — 탐색용)

### 날짜

2026-10-12

### Exp-R4: GT 물체가 어느 단계에서 사라지는가 (FS L&F, 원본 bar)

`src/diag_small_object_loss.py` — IE-NMS 이전 후보(ID top-300 + UN 300)로 GT component마다 첫 실패 단계를 판정한다. 기준: box IoU ≥ 0.5, GT ≥ 10px.

| GT 면적 | n | 후보 없음 | ID label로만 | 점수 ≤ 0.2 | NMS 제거 | 검출 |
|---|---|---|---|---|---|---|
| <16² | 60 | **27 (45%)** | 0 | **25 (42%)** | 1 (2%) | 7 (12%) |
| 16²~32² | 56 | 3 (5%) | 0 | **19 (34%)** | 3 (5%) | 31 (55%) |
| 32²~96² | 56 | 0 | 0 | 7 (12%) | 5 (9%) | 44 (79%) |
| ≥96² | 16 | 0 | 0 | 2 (12%) | 2 (12%) | 12 (75%) |
| 전체 | 188 | 30 (16%) | 0 | 53 (28%) | 11 (6%) | 94 (50%) |

- 위치를 맞게 잡은 label-0 후보의 최고 점수 중앙값: <16² **0.165**, 16²~32² **0.234**, 32²~96² 0.321, ≥96² 0.389 → **작을수록 objectness가 낮다**
- 후보 box의 위치 정확도는 높다 (가장 잘 맞는 후보의 IoU 중앙값: 16²~32² 0.87, <16² 0.59)
- NMS 제거는 소수(6%). thing bar 0.2에서는 10%로 늘어난다 (주로 boosted motorcycle)
- 결론: 작은 물체 손실의 주원인은 **(1) 최종 threshold 0.2에 못 미치는 낮은 점수**와 **(2) 16px 미만에서 후보 자체가 없음**(입력이 2048→1333으로 줄어드는 영향으로 추정, 확인 필요)이다. IE-NMS는 작은 물체 손실의 주원인이 아니다

### Exp-E2: size-aware threshold (면적 < 32² 인 box는 score > 0.1로 유지)

Box 수준(오프라인, thing bar 0.2 기준):

| | box/img | FP/img | recall | <16² | 16²~32² | ≥32² |
|---|---|---|---|---|---|---|
| FS L&F thing .2 | 4.61 | 3.69 | 45.7 | 10.0 | 51.8 | 70.8 |
| FS L&F thing .2 + small .1 | 7.53 | 6.19 | **60.6** | **41.7** | **67.9** | 70.8 |
| (FS L&F 원본) | 10.51 | 9.51 | 50.0 | – | – | – |
| RA thing .2 | 7.92 | 4.38 | 53.8 | 16.0 | 57.9 | 77.7 |
| RA thing .2 + small .1 | 11.25 | 7.42 | 57.5 | 24.7 | 63.2 | 77.7 |

Mask 수준 (DetSeg-S + SAM):

| config | FS IoU | FS F1 | FS 전체 픽셀 IoU | FS IoU=0 이미지 | FS IoU 중앙값 | FS box recall | RA IoU |
|---|---|---|---|---|---|---|---|
| 원본 | 14.26 | 18.53 | 6.19 | 44 | 0.68 | 50.0 | 76.46 |
| bar(thing .2) | **20.96** | 26.19 | **26.63** | 52 | 0.00 | 45.7 | **78.68** |
| size-aware만 | 14.07 | 18.66 | 6.24 | **25** | 1.17 | **64.9** | 74.52 |
| bar + size-aware | 20.57 | **27.29** | 25.37 | 33 | **7.71** | 60.6 | 76.49 |

### 관찰

1. **bar 수정은 두 dataset 모두에서 mask IoU를 올린다** (FS +6.7, RA +2.2). 가장 확실한 효과
2. **size-aware는 물체 단위로는 크게 좋아진다**: FS 작은 물체 box recall 30.2 → 54.3, 0점 이미지 52 → 33, IoU 중앙값 0 → 7.7. 원본보다 recall은 높고(60.6 vs 50.0) FP는 적다(6.19 vs 9.51)
3. **그런데 평균 mask IoU는 오르지 않는다** (20.96 → 20.57). 작은 물체를 새로 잡아도 이미지별 IoU 이득이 작고, 추가된 작은 FP가 다른 이미지의 IoU를 깎기 때문으로 보인다 (확인 필요)
   → **이미지 평균 IoU는 "작은 장애물을 놓치지 않는가"를 잘 반영하지 못하는 지표다.** 작은 물체 개선을 주장하려면 물체 단위 지표(component recall/precision, SMIYC식 sIoU/PPV/F1)가 필요하다
4. Road Anomaly에서는 size-aware가 IoU를 약간 낮춘다 (−1.9 단독, 결합 시 bar 단독 대비 −2.2). 작은 FP 증가의 대가
5. 남은 FP를 줄이는 장치(road-aware)가 size-aware와 짝을 이뤄야 한다는 근거가 된다

### 다음 작업

- 물체 단위 지표를 평가에 추가 (component recall / precision, 가능하면 SMIYC식 sIoU·PPV·F1)
- road-aware: size-aware로 늘어난 작은 FP 중 도로 밖에 있는 비율을 먼저 측정
- 16px 미만 "후보 없음" → 고해상도 추론(FixScaleResize 상향) 효과 확인
- 최종 수치는 보정 데이터로 bar와 threshold를 정한 뒤 보고

---

## 물체 단위 지표 + road-aware 결합 실험 (Exp-E3, 탐색용)

### 날짜

2026-10-13

### 주의

탐색 실험이다. 모든 하이퍼파라미터(thing bar 0.2, small 기준 32²·0.1, road α 0.5)를 FS L&F / Road Anomaly 결과를 보며 정했다. **최종 보고 전에 이 값을 고정하고, 한 번도 보지 않은 데이터에서 검증해야 한다.**

### 추가한 것

1. **물체 단위 지표** (`src/eval_detseg_s_diagnostics.py`의 `component_metrics`): SMIYC(Chan et al., 2021) 방식을 근사 구현
   - sIoU_gt: GT 물체별로 맞닿은 예측 영역과의 IoU (다른 GT 픽셀은 제외), 물체 평균
   - PPV: 예측 영역별로 GT 물체 위에 있는 비율, 예측 평균
   - mean F1: τ ∈ {0.25, …, 0.75}에서 TP = sIoU>τ인 GT, FN = 나머지 GT, FP = PPV≤τ인 예측으로 F1을 구해 평균
   - void(255)는 예측 영역에서 제외, 10px 미만 GT 물체는 무시. 합성 데이터로 손계산 결과와 일치하는지 확인함
2. **road-aware 재점수화**: 별도 모델 없이 **detector 자신의 'road' box**(label 1, score>0.2, NMS 이전)를 도로 영역으로 쓴다. box의 바닥 중앙점이 road box 밖이면
   - soft: score × α (α=0.5)
   - hard_small: 작은 box(<32²)만 제거
   - road box가 없는 이미지는 건드리지 않는다
   - 사전 분석: FS L&F에서 실제 장애물 GT의 99%, Road Anomaly에서 90%가 road box 위에 있다. 반면 작은 FP는 55% / 48%만 road box 위에 있다. 사다리꼴 위치 prior는 TP를 28~42% 잃어 제외했다

### 결과 (DetSeg-S + SAM)

FS L&F:

| config | IoU | 전체 픽셀 IoU | 0점 이미지 | sIoU | PPV | mean F1 | box recall | 작은(<32²) recall |
|---|---|---|---|---|---|---|---|---|
| 원본 | 14.26 | 6.19 | 44 | 43.25 | 10.31 | 17.81 | 50.0 | 32.8 |
| + bar | 20.96 | 26.63 | 52 | 39.59 | 18.29 | 26.95 | 45.7 | 30.2 |
| + size | 14.07 | 6.24 | 25 | **54.55** | 10.67 | 19.06 | **64.9** | **56.9** |
| + bar + size | 20.57 | 25.37 | 33 | 50.89 | 15.88 | 25.96 | 60.6 | 54.3 |
| + bar + road(soft) | 24.71 | **28.11** | 52 | 39.10 | **23.22** | 31.57 | 45.7 | 30.2 |
| **+ bar + size + road(soft)** | **26.60** | 27.43 | **33** | 50.40 | 20.77 | **31.74** | 60.6 | 54.3 |
| + bar + size + road(hard_small) | 23.03 | 26.76 | 33 | 50.89 | 18.95 | 29.72 | 60.6 | 54.3 |
| (논문 Table 2) | 32.36 | | | | | | | |

Road Anomaly:

| config | IoU | 전체 픽셀 IoU | sIoU | PPV | mean F1 | box recall | 작은 recall |
|---|---|---|---|---|---|---|---|
| 원본 | 76.46 | 76.85 | 54.61 | 47.73 | 46.95 | 47.0 | 23.2 |
| + bar | **78.68** | **80.65** | 52.40 | 54.22 | 50.39 | 43.3 | 19.8 |
| + size | 74.52 | 76.32 | **56.83** | 38.28 | 40.30 | **50.0** | **28.2** |
| + bar + size | 76.49 | 79.99 | 54.64 | 40.74 | 41.33 | 46.3 | 24.9 |
| + bar + road(soft) | 78.48 | 76.99 | 51.19 | **60.34** | **53.71** | 43.0 | 19.8 |
| **+ bar + size + road(soft)** | 77.08 | 76.64 | 53.36 | 49.66 | 47.52 | 46.0 | 24.9 |
| + bar + size + road(hard_small) | 78.12 | 80.52 | 54.60 | 50.07 | 48.31 | 46.3 | 24.9 |
| (논문 Table 2) | 68.35 | | | | | | |

### 관찰

1. **전체 결합(bar + size + road soft)이 FS L&F에서 모든 주요 지표를 원본보다 크게 개선**: IoU 14.26 → 26.60, mean F1 17.81 → 31.74, PPV 10.31 → 20.77, sIoU 43.25 → 50.40, 작은 물체 recall 32.8 → 54.3, 0점 이미지 44 → 33
   - 논문 32.36과의 차이: 18.1 → 5.8 (판정 기준상 "부분 재현" 범위)
2. **각 요소의 역할이 지표에서 분리된다** (ablation이 성립):
   - bar → FP 제거 (PPV↑, IoU↑)
   - size-aware → 작은 물체 회수 (sIoU↑, recall↑, 0점 이미지↓), 대신 PPV↓
   - road-aware → size-aware가 늘린 FP를 다시 줄임 (PPV↑, F1↑)
3. Road Anomaly에서는 전체 결합이 원본보다 소폭 좋지만(IoU +0.6, F1 +0.6, PPV +1.9, sIoU −1.3), **bar + road(size 없이)가 더 좋다** (F1 53.71). 물체가 크고 도로 밖 이상물체가 있는 dataset에서는 size-aware의 이득이 작고 FP 비용이 크다 → dataset 성격에 따른 trade-off로 보고
4. road는 soft가 FS에서, hard_small이 RA에서 약간 낫다. 한 가지로 고정해야 한다 (soft 권장: 두 dataset 모두 원본 대비 개선)

### 다음 작업

- 하이퍼파라미터 고정 → **한 번도 보지 않은 held-out 데이터**(SMIYC Obstacle val 30장, SMIYC Anomaly val 10장)에서 검증
- 가능하면 bar를 보정 데이터로 정하는 절차 구현 (논문 방식)

---

## 재현 실패 원인 재점검: 공식 repo 코드 이력 분석 (Exp-R5)

### 날짜

2026-10-13

### 방법

`external/DetSeg-official`의 git 이력을 확인하고, refactor 이전 코드(`git show 7953dc8e^:mmdet/detseg_utils/models.py`, 2889줄)와 현재 코드(1769줄)를 비교했다.

### 커밋 이력

| 날짜 | 커밋 | 내용 |
|---|---|---|
| 2024-10 ~ 2025-03 | 9a68606d ~ ff273121 | 연구 중 config·model 업로드 |
| 2025-10 | – | 논문 발표 (ICCV 2025) |
| 2026-02-25 | 2c39e4bf | "upload code": 연구 당시 코드 전체. 가중치 없음 |
| 2026-06-23 | – | 공개 checkpoint 학습일 (파일명 `detseg_swin-b_coco_20260623-207453f5.pth`) |
| 2026-07-03 | 7953dc8e | **"Refactor DetSeg modules"**: models.py 2842줄 변경 |
| 2026-07-03 | b5337107 | 공개 checkpoint 추가 |
| 2026-09-29 | a1aca17b | SMIYC 제출 도구 등 (현재 HEAD) |

### 연구 당시 코드 vs 공개 코드

| 항목 | 연구 당시 (2c39e4bf) | 공개 (a1aca17b) |
|---|---|---|
| 가중치 | 논문 실험 가중치 (비공개) | 2026-06-23 재학습 가중치 |
| decoder | UN query 전용 추가 layer `bbyy_layer1` (`GroundingDinoTransformerDecoderPT`) | 없음 (표준 decoder) |
| IE-NMS bar | `thres_dict` {1–11: 0.2, 12–19: 0.5}, ×2 강화, IoU 0.5 | 동일 (`[0.2]*11 + [0.5]*8`) |
| SAM | `./sam-vit-base` | `facebook/sam-vit-base` (동일 크기) |
| text attention mask | `query_text_mask` 생성만 하고 미적용 | 동일 |
| 작은 box 재점수화 | `GroundingDINOPTDetSegSAM`: 면적 < 1024이고 score > 0.1인 box의 score를 **seg branch anomaly score의 box 평균(+0.3)으로 교체** | 없음 |
| DetSeg-S 계열 | `GroundingDINOPT`(score>0.2 & label 0 + SAM), `GroundingDINOPTDetSegSAM`(det+seg, 위 재점수화), `GroundingDINOPTSegSAM`(label 0만, score 필터 주석) 등 여러 변형. 주석 처리된 threshold 0.25 / 0.35 존재 | `DetSegS` 하나 (score>0.2 & label 0 + SAM) |
| 원래 DetSeg-S config | `grounding_dino_swin-b_detseg_sam_1ldl.py` → `GroundingDINOPTDetSegSAM` | `detseg-s_swin-b.py` → `DetSegS` |

### README 근거

- "Small differences from the paper numbers are expected because the released checkpoint is retrained with the same config and subject to normal training variance."
- 공개 checkpoint 결과표에는 **DetSeg-R만 있고 DetSeg-S는 없다**
- 그 표에서도 FS L&F가 두드러지게 나빠졌다: M2A AP 78.15 → 72.54, FPR95 4.84 → 9.40 / UNO FPR95 1.95 → 5.11 / RbA FPR95 6.53 → 9.80

### 결론

1. 논문 수치를 만든 **코드와 가중치는 공개되지 않았다.** 공개본은 논문 약 9개월 후 재작성된 코드와 재학습된 가중치다
2. IE-NMS bar(0.2/0.5)는 원래 가중치 시절 값이 그대로 남았고, 재학습 가중치의 ID 점수 분포(중앙값 0.26)와 맞지 않아 IE-NMS가 작동하지 않는다 (Exp-R3과 일치)
3. 연구 당시 코드에는 논문에 기술되지 않은 작은 box 재점수화가 있었고 공개본에서 제거되었다. 어떤 코드 경로로 Table 2를 산출했는지는 공개 자료로 확정할 수 없다
4. 저자도 공개 checkpoint로 DetSeg-S 수치를 보고하지 않았다

### 보고서 서술 시 주의

- 개선 ①(bar 수정)은 "공개본의 bar를 재학습 가중치에 맞게 바로잡은 것"으로 서술한다
- 개선 ②(size-aware)는 원래 코드에도 유사한 아이디어(면적 < 1024, score > 0.1)가 있었다. "최초 고안"으로 쓰지 않는다. 차이점: seg branch 없이 동작, road-aware와 결합, 진단을 근거로 도출
- 비교 기준은 논문 수치가 아니라 공개본 원본 실행 결과(FS L&F 14.26, RA 76.46)다
