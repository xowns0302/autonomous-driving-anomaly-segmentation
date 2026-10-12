# DetSeg 재현·진단·개선 실험 기록 (보고서 작성용)

- 기간: 2026-10-12 ~ 2026-10-13
- 대상: DetSeg (Zhu et al., *Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes*, ICCV 2025) 공식 공개본
- 이 문서의 역할: 보고서·발표를 쓸 때 필요한 **모든 설정, 수치, 해석, 재현 방법, 근거 파일**을 한곳에 모은다
  - 실험별 원자료와 세부 표: [`experiments/result_log.md`](../experiments/result_log.md) (Exp-R0 ~ R5, E1 ~ E3)
  - 코드 분석: [06](06_detseg_code_analysis.md) · 방향: [07](07_project_direction_after_code_analysis.md) · 재현 계획: [08](08_detseg_s_baseline_reproduction_plan.md)
  - 시각화 정리 페이지: [`reports/detseg_experiment_log.html`](../reports/detseg_experiment_log.html)
- **현재 개선 수치는 모두 탐색용이다** (설정값을 평가 데이터를 보면서 골랐다). 최종 보고 수치는 held-out 검증 뒤에 확정한다 (9절)

---

## 1. 한 문단 요약

공개된 DetSeg 코드와 checkpoint를 수정 없이 실행한 결과, DetSeg-S의 FS Lost&Found IoU는 14.26으로 논문(32.36)의 44%에 그쳤고, Road Anomaly는 76.46으로 논문(68.35)보다 높았다. 이미지별·물체별 진단을 통해, FS L&F에서 예측 box의 90.5%가 정답과 무관하며 그 대부분이 일반 차량 위에 있음을 확인했다. IE-NMS 이전 단계를 추적한 결과, ID branch는 차량을 올바른 class로 검출했지만 점수(중앙값 0.26)가 코드에 하드코딩된 ID 판정 기준(0.5)을 넘지 못해 IE-NMS가 한 번도 작동하지 않았다. 공식 저장소의 커밋 이력을 분석한 결과, 공개 코드는 논문 발표 약 9개월 뒤 재작성되었고 checkpoint도 재학습된 것이었으며, 기준값은 원래 가중치 시절 값이 그대로 남아 있었다. 또한 작은 장애물은 objectness 점수가 낮아 최종 threshold에서 탈락하거나(16~32px의 34%), 해상도 부족으로 후보조차 생성되지 않았다(16px 미만의 45%). 이 진단에 따라 재학습 없는 후처리 세 가지(① IE-NMS 기준값 수정, ② 크기 적응 threshold, ③ detector 자신의 road 예측을 이용한 도로 문맥 재점수화)를 적용한 결과, FS L&F에서 IoU 14.26 → 26.60, 물체 단위 mean F1 17.81 → 31.74, 작은 물체 box 검출률 32.8% → 54.3%를 얻었고, Road Anomaly에서도 원본 대비 소폭 개선(IoU 76.46 → 77.08)되었다.

---

## 2. 실험 환경 (재현 정보)

| 항목 | 값 |
|---|---|
| OS | WSL2 Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 4070 Ti (12GB) |
| conda env | `detseg` |
| Python | 3.8.20 |
| PyTorch | 2.1.0+cu118 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| mmdet (repo 내장) / mmsegmentation | 3.3.0 / 1.2.2 |
| transformers | 4.46.3 (SAM은 HuggingFace `facebook/sam-vit-base`) |
| 공식 repo | `external/DetSeg-official`, HEAD `a1aca17b` (2026-09-29) |
| checkpoint | `ckpts/detseg_swin-b_coco_20260623-207453f5.pth` (README: 재학습본) |
| DetSeg-S config | `configs/detseg/detseg-s_swin-b.py` (원본 그대로) |
| Road Anomaly config | `experiments/configs/detseg-s_road_anomaly.py` (`RoadAnomalyDataset` + `UnifyGT({0:0, 2:1})`) |
| 입력 resize | `FixScaleResize(800, 1333, keep_ratio)` → 2048×1024 이미지는 약 1333×666 |
| 추론 속도 | 약 1초/장 (SAM 포함) |
| 원본 코드 수정 | **없음.** 모든 변경은 우리 스크립트에서 `_ie_nms`를 교체(monkeypatch)해 적용 |

검증 사항:
- checkpoint 로드 시 missing/unexpected key 경고 없음
- 배포 checkpoint의 파라미터 그룹: backbone, neck, encoder, decoder, bbox_head, language_model, query_embedding, universal_* 등. **`seg_decoder`, `neck_seg` 없음**
- 우리 진단 스크립트 결과가 `tools/test.py` 공식 결과와 일치 (FS L&F 14.26)
- 교체한 `_ie_nms`에 원래 값(0.5)을 넣으면 14.26이 그대로 재현됨

---

## 3. 데이터셋

모두 `~/datasets/`에 두고 `external/DetSeg-official/data/`에 symlink.

| Dataset | 장수 | 해상도 | GT 값 | OoD 없는 이미지 | 출처 | DetSeg 경로 | 상태 |
|---|---|---|---|---|---|---|---|
| FS L&F | 100 | 2048×1024 | 0, 1, 255 | 1 | Mask2Anomaly `Validation_Dataset/FS_LostFound_full` (Google Drive) | `data/FS_LostFound` | 재현·탐색 사용 |
| Road Anomaly | 60 | 1280×720 | 0, **2** | 0 | EPFL 공식 `RoadAnomaly.zip` (WebP) | `data/RoadAnomaly` | 재현·탐색 사용 |
| FS Static | 30 | 2048×1024 | 0, 1, 255 | **10** | Mask2Anomaly 묶음 `fs_static` | `data/FS_Static` | 보류 |
| SMIYC Obstacle val | 30 | 1920×1080 | 0, 1, 255 | 0 | Mask2Anomaly 묶음 `RoadObsticle21` | (미연결) | **held-out, 아직 안 봄** |
| SMIYC Anomaly val | 10 | 1280×720 | 0, 1, 255 | 0 | Mask2Anomaly 묶음 `RoadAnomaly21` | (미연결) | **held-out, 아직 안 봄** |

다운로드:
- Road Anomaly: `wget https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly.zip` (13MB)
- Mask2Anomaly 묶음: https://drive.google.com/drive/folders/1eQhmPbKSZrN1AsieY9KFchfll7XC1_SF 의 `Validation_Dataset.zip` (329MB). 같은 폴더에 M2A 가중치 `best_contrastive.pth`도 있음 (DetSeg-R external 용 score map 생성에 쓸 수 있음)

주의:
- **Road Anomaly**는 anomaly가 2로 라벨링되어 있다. DetSeg-S 지표(`BinaryMaskAnomalyIoUMetric`)는 `gt == 1`만 OoD로 보므로 `UnifyGT({0:0, 2:1})`가 필수다 (config에는 주석 처리되어 있음)
- **FS Static**(이 버전)은 30장 중 10장에 OoD가 없다. 이미지 평균 IoU에서 이 10장은 항상 0점이라 최대 66.7까지만 나온다. 논문 83.34는 다른 버전으로 추정된다 (논문 4.1절도 FS Static 버전 불일치를 언급, 공식 평가 스크립트는 FS Static만 `original/labels` 폴더 구조를 씀)
- FS L&F 100장은 LostAndFound 원본에서 뽑은 것이다. LostAndFound 원본을 튜닝용으로 쓸 경우 이미지 중복을 확인해야 한다

---

## 4. 평가 지표 정의

| 지표 | 정의 | 구현 |
|---|---|---|
| **IoU (공식)** | 이미지마다 IoU = \|GT∩Pred\| / \|(GT∪Pred) \ void\|를 구해 **평균**. GT도 Pred도 없으면 0점 | `BinaryMaskAnomalyIoUMetric` (`metrics.py:680-700`), 우리 스크립트도 동일 |
| F1 (공식) | 이미지마다 2I/(I+U) (= Dice) 평균 | 동일 |
| 전체 픽셀 IoU | 모든 이미지의 I와 U를 합산한 뒤 나눔 | `eval_detseg_s_diagnostics.py` |
| box recall | GT 연결요소의 외접 box와 예측 box의 IoU ≥ 0.5이면 검출. 면적 구간 <32², 32²~96², ≥96² (일부 분석은 <16², 16²~32² 세분) | 같은 스크립트 |
| FP-like box | 어떤 GT 연결요소와도 box IoU < 0.1인 예측 box | `src/analysis/fp_analysis.py` 등 |
| **sIoU** | GT 물체 k마다 \|k ∩ P(k)\| / \|(k ∪ P(k)) \ A(k)\|. P(k)=k에 닿은 예측 연결요소, A(k)=다른 GT 픽셀. 물체 평균 | `component_metrics()` (SMIYC, Chan et al. 2021 근사) |
| **PPV** | 예측 연결요소 p마다 \|p ∩ K(p)\| / \|p\|. K(p)=p에 닿은 GT. 예측 평균 | 같은 함수 |
| **mean F1** | τ ∈ {0.25, 0.30, …, 0.75}마다 TP=sIoU>τ인 GT 수, FN=나머지 GT, FP=PPV≤τ인 예측 수로 F1을 구해 평균 | `summarize_components()` |

- 물체 단위 지표에서 void(255) 픽셀은 예측에서 제외, 10px 미만 GT 물체는 무시 (Road Anomaly의 1px 라벨 잡음 때문)
- 합성 데이터로 손계산 결과와 일치함을 확인 (sIoU 100/150, F1@0.5 = 2/(2+1+1))
- **이미지 평균 IoU는 작은 장애물 개선을 잘 반영하지 못한다** (E2 참고). 작은 물체 관련 주장은 물체 단위 지표로 한다

판정 기준 (실행 전 고정, 08 문서): 논문과 차이 ≤ 3 재현됨 · 3~10 부분 재현 · > 10 불일치

---

## 5. 실험 목록

| ID | 날짜 | 종류 | 질문 | 핵심 결과 | 결과 파일 |
|---|---|---|---|---|---|
| R0 | 10-12 | sanity | 로컬에서 돌아가나 | box는 유효, heatmap은 무의미 (seg branch 가중치 없음) | `assets/qualitative_results/` |
| R1 | 10-12 | 재현 | FS L&F가 논문 수치(32.36)를 재현하나 | **14.26, 불일치** | `experiments/results/repro_detseg_s/` |
| R2 | 10-12 | 진단 | 왜 낮은가, RA는 어떤가 | FS: 예측 box 90.5%가 정답과 무관, 대부분 차량 위. RA 76.46 (논문보다 높음) | `experiments/results/diag_detseg_s/` |
| R3 | 10-12 | 진단 | 차량 box는 왜 IE-NMS에서 안 지워지나 | ID box 점수(중앙 0.26)가 기준 0.5를 한 번도 못 넘음 → IE-NMS 미작동 | `experiments/results/diag_ie_nms/*/final_boxes.json` |
| E1 | 10-12 | 탐색 | 기준값을 낮추면 | thing 0.2: FS 14.26→20.96, RA 76.46→78.68 | `experiments/results/sweep_iou/` |
| R4 | 10-12 | 진단 | 작은 장애물은 어디서 사라지나 | 점수 미달 + 16px 미만 후보 없음. IE-NMS는 주원인 아님 | (오프라인, candidates npz) |
| E2 | 10-12 | 탐색 | 작은 box 기준을 낮추면 | 작은 물체 검출률 30→54%, 평균 IoU는 그대로 | `experiments/results/sweep_iou/` |
| E3 | 10-13 | 탐색 | 물체 단위 지표 + road-aware 결합 | FS IoU 26.60, mean F1 31.74 / RA IoU 77.08 | `experiments/results/comp_eval/` |
| R5 | 10-13 | 진단 | 공식 코드로 왜 재현이 안 되나 | 공개본은 논문 9개월 뒤 재작성 + 재학습. 기준값은 옛 가중치 기준 | git 이력 (result_log R5) |

---

## 6. 실험별 상세

### R0. 단일 이미지 sanity check

- 단일 이미지 추론 시 `KeyError: 'text'` → `ConcatPrompt` 앞에 Cityscapes 19 class prompt를 넣는 transform 추가 (`src/run_detseg_single_image.py`)
- internal config는 `seg_decoder`로 heatmap을 만드는데 배포 checkpoint에 해당 가중치가 없다 → heatmap(값 −18.8 ~ −13.1)은 랜덤 초기화 출력 + box 내부 +1.0. **해석 대상에서 제외**
- road debris 이미지 threshold 0.20/0.30/0.50 비교는 모델 내부 0.2 필터 이후의 시각화용이라 0.2 미만은 효과 없음

### R1. FS L&F baseline 재현

| Dataset | 장수 | 논문 IoU | 우리 IoU | 차이 | F1 | 판정 |
|---|---|---|---|---|---|---|
| FS L&F | 100/100 | 32.36 | 14.26 | −18.10 | 18.53 | 불일치 |

### R2. 재현 실패 진단

| 지표 | FS L&F | Road Anomaly |
|---|---|---|
| IoU (공식) / 논문 | 14.26 / 32.36 | 76.46 / 68.35 |
| 전체 픽셀 IoU | 6.19 | 76.85 |
| 이미지당 box 수 | 10.5 | 10.2 |
| 정답과 무관한 box 비율 | **90.5%** (951/1051) | 62.4% (382/612) |
| 예측 면적 / 정답 면적 (중앙값) | **47.4배** | 1.06배 |
| IoU 0점 이미지 | **44/100** | 0/60 |
| IoU 분위수 p25 / p50 / p75 / p90 | 0 / 0.68 / 17.31 / 56.53 | 68.72 / 85.11 / 92.65 / 96.48 |
| 정답 물체 면적 중앙값 | 664px | 427px (잡음 포함) |
| box score 중앙값 FP / TP | 0.343 / 0.327 | 0.309 / 0.470 |

- 그림: `assets/qualitative_results/baseline_detseg_s/fs_lostfound_fp_examples.jpg` (FS L&F #0, #10, #30, #55. 초록=예측, 빨강=정답). 오탐은 승용차·트럭·밴·자전거·자차 보닛 위
- 해석: 원인은 같다(차량 위 오탐). FS L&F는 정답이 작아서 오탐 몇 개가 union을 지배하고, RA는 물체가 커서 영향이 작다. 오탐과 정답 점수가 비슷해 threshold로는 분리되지 않는다

### R3. IE-NMS 출처 추적 (FS L&F)

| 항목 | 전체 1051 | 오탐 951 | 정답 100 |
|---|---|---|---|
| UN branch 출처 | 926 | 876 (92%) | 50 |
| ID branch 'objects' label 출처 | 125 | 75 (8%) | 50 |
| ID class box와 IoU≥0.5로 겹침 | 910 (86.6%) | 860 (90.4%) | 50 (50%) |
| 겹친 ID box가 기준 통과 | 0 | 0 | 0 |
| 겹친 ID box 점수 p10/p50/p90 | 0.12/0.25/0.45 | 0.13/0.26/0.45 | 0.10/0.12/0.39 |
| 겹친 ID box class | | car 439, person 86, truck 69, pole 56, bicycle 55, bus 47, motorcycle 37, fence 29 | pole 12, motorcycle 10, traffic sign 9, fence 5, person 5 |

- 결론: (a) "ID box 점수가 기준 미달이라 강화 안 됨"이 주원인. (b) "attention mask 미적용으로 ID branch가 차를 'objects'로 분류"는 8%

### E1. IE-NMS 기준값 스윕 (탐색)

box 수준 (오프라인, `src/sweep_ie_nms_bars.py`):

| 설정 | FS FP/img | FS recall | FS <32² | FS ≥96² | RA FP/img | RA recall |
|---|---|---|---|---|---|---|
| 원본 (stuff .2 / thing .5) | 9.51 | 50.0 | 32.8 | 75.0 | 6.37 | 58.3 |
| thing .4 | 7.64 | 47.3 | 31.9 | 56.2 | 5.80 | 57.9 |
| thing .3 | 5.59 | 46.8 | 31.9 | 56.2 | 5.25 | 55.0 |
| thing .2 | 3.69 | 45.7 | 30.2 | 56.2 | 4.38 | 53.8 |
| thing .15 | 2.75 | 45.7 | 30.2 | 56.2 | 3.90 | 52.1 |
| thing .1 | 2.34 | 44.1 | 28.4 | 56.2 | 3.73 | 52.1 |
| all .1 | 1.71 | 41.5 | 25.9 | 56.2 | 3.30 | 52.1 |
| thing .2 + 장애물형 class 보호 | 3.91 | 47.3 | 31.0 | 56.2 | 4.45 | 53.8 |
| IE-NMS 강화 없음 | 14.77 | 51.6 | 33.6 | 75.0 | 7.57 | 58.3 |

mask IoU: 원본 14.26 / 76.46 → **thing .2: 20.96 / 78.68** → thing .2 + 보호: 20.53 / 77.55 (FS / RA)

- 장애물형 class(pole, traffic light, traffic sign, fence) 보호는 **효과 없음 → 기각**

### R4. 작은 장애물 손실 단계 (FS L&F, 원본 기준값, GT ≥ 10px)

| GT 면적 | n | 후보 없음 | ID label로만 | 점수 ≤ 0.2 | NMS 제거 | 검출 | 최고 OoD 후보 점수 중앙값 |
|---|---|---|---|---|---|---|---|
| <16² | 60 | 27 (45%) | 0 | 25 (42%) | 1 (2%) | 7 (12%) | 0.165 |
| 16²~32² | 56 | 3 (5%) | 0 | 19 (34%) | 3 (5%) | 31 (55%) | 0.234 |
| 32²~96² | 56 | 0 | 0 | 7 (12%) | 5 (9%) | 44 (79%) | 0.321 |
| ≥96² | 16 | 0 | 0 | 2 (12%) | 2 (12%) | 12 (75%) | 0.389 |
| 전체 | 188 | 30 (16%) | 0 | 53 (28%) | 11 (6%) | 94 (50%) | |

- 최고 후보 box의 GT IoU 중앙값: <16² 0.59, 16²~32² 0.87 → 위치는 잘 잡는다
- thing 0.2에서는 NMS 제거가 10%로 늘어남 (주로 강화된 motorcycle box)
- localized 기준을 IoU≥0.3으로 완화해도 경향 동일
- **"IE-NMS가 작은 장애물을 지운다" 가설 기각**

### E2. size-aware threshold (탐색)

면적 < 32² box는 최종 기준 0.2 대신 0.1 (구현: IE-NMS 뒤 작은 box 점수 ×2, DetSeg-S는 box만 쓰므로 동치)

| (thing .2 기준) | FS FP/img | FS recall | <16² | 16²~32² | ≥32² | RA FP/img | RA recall |
|---|---|---|---|---|---|---|---|
| small 기준 없음 | 3.69 | 45.7 | 10.0 | 51.8 | 70.8 | 4.38 | 53.8 |
| < 32² → 0.15 | 4.82 | 53.7 | 25.0 | 62.5 | 70.8 | 5.55 | 55.4 |
| **< 32² → 0.1** | 6.19 | 60.6 | 41.7 | 67.9 | 70.8 | 7.42 | 57.5 |
| < 48² → 0.15 | 5.46 | 56.4 | 25.0 | 66.1 | 75.0 | 6.18 | 56.2 |
| < 48² → 0.1 | 8.03 | 63.3 | 41.7 | 71.4 | 75.0 | 9.50 | 58.8 |

- mask 수준: 평균 IoU는 거의 변화 없음 (FS 20.96 → 20.57), 그러나 0점 이미지 52 → 33, IoU 중앙값 0 → 7.71

### E3. 물체 단위 지표 + road-aware (탐색)

road-aware 사전 분석 (thing .2 + small .1의 최종 box, 바닥 중앙점 기준):

| | FS: GDINO road box 위 | FS: 사다리꼴 prior 안 | RA: road box 위 | RA: 사다리꼴 안 |
|---|---|---|---|---|
| 정답 box (TP) | 99% | 72% | 88% | 81% |
| 오탐 box (FP) | 64% | 18% | 49% | 34% |
| 작은 정답 box | 100% | 58% | 90% | 62% |
| 작은 오탐 box | 55% | 7% | 48% | 25% |
| GT 물체 | 99% | 66% | 90% | 82% |

road 규칙 box 수준 비교 (thing .2 + small .1 기준):

| road 규칙 | FS FP/img | FS recall | RA FP/img | RA recall | RA ≥32² recall |
|---|---|---|---|---|---|
| 없음 | 6.19 | 60.6 | 7.42 | 57.5 | 77.7 |
| hard (도로 밖 제거) | 3.94 | 60.6 | 3.92 | 55.0 | 72.7 |
| **soft ×0.5** | 4.29 | 60.6 | 4.48 | 57.1 | 76.9 |
| soft ×0.7 | 5.03 | 60.6 | 5.53 | 57.5 | 77.7 |
| hard (작은 box만) | 4.68 | 60.6 | 5.35 | 57.5 | 77.7 |

최종 ablation (DetSeg-S + SAM):

FS L&F

| 설정 | IoU | 전체 IoU | 0점 이미지 | sIoU | PPV | mean F1 | box recall | <32² recall | 예측 물체 수 |
|---|---|---|---|---|---|---|---|---|---|
| 원본 | 14.26 | 6.19 | 44 | 43.25 | 10.31 | 17.81 | 50.0 | 32.8 | 852 |
| ① bar | 20.96 | 26.63 | 52 | 39.59 | 18.29 | 26.95 | 45.7 | 30.2 | 441 |
| ② size | 14.07 | 6.24 | 25 | 54.55 | 10.67 | 19.06 | 64.9 | 56.9 | 1041 |
| ①+② | 20.57 | 25.37 | 33 | 50.89 | 15.88 | 25.96 | 60.6 | 54.3 | 654 |
| ①+③ road soft | 24.71 | 28.11 | 52 | 39.10 | 23.22 | 31.57 | 45.7 | 30.2 | 343 |
| **①+②+③ road soft** | **26.60** | 27.43 | 33 | 50.40 | 20.77 | **31.74** | 60.6 | 54.3 | 495 |
| ①+②+③ road hard_small | 23.03 | 26.76 | 33 | 50.89 | 18.95 | 29.72 | 60.6 | 54.3 | 548 |
| 논문 Table 2 | 32.36 | | | | | | | | |

Road Anomaly

| 설정 | IoU | 전체 IoU | sIoU | PPV | mean F1 | box recall | <32² recall | 예측 물체 수 |
|---|---|---|---|---|---|---|---|---|
| 원본 | 76.46 | 76.85 | 54.61 | 47.73 | 46.95 | 47.0 | 23.2 | 454 |
| ① bar | 78.68 | 80.65 | 52.40 | 54.22 | 50.39 | 43.3 | 19.8 | 391 |
| ② size | 74.52 | 76.32 | 56.83 | 38.28 | 40.30 | 50.0 | 28.2 | 583 |
| ①+② | 76.49 | 79.99 | 54.64 | 40.74 | 41.33 | 46.3 | 24.9 | 539 |
| ①+③ road soft | 78.48 | 76.99 | 51.19 | 60.34 | 53.71 | 43.0 | 19.8 | 342 |
| **①+②+③ road soft** | 77.08 | 76.64 | 53.36 | 49.66 | 47.52 | 46.0 | 24.9 | 433 |
| ①+②+③ road hard_small | 78.12 | 80.52 | 54.60 | 50.07 | 48.31 | 46.3 | 24.9 | 439 |
| 논문 Table 2 | 68.35 | | | | | | | |

(box recall은 GT ≥ 10px 기준이 아니라 전체 GT 연결요소 기준이라 E1·E2 표와 RA 수치가 다르다)

### R5. 공식 코드 이력 분석

- 2026-02-25 연구 당시 코드 업로드(가중치 없음) → 2026-06-23 checkpoint 재학습 → 2026-07-03 "Refactor DetSeg modules"(models.py 2842줄 변경) + checkpoint 공개
- 연구 당시 코드와 비교: decoder에 UN 전용 layer(`bbyy_layer1`) 있었음 / IE-NMS 기준값 동일 / SAM ViT-B 동일 / text mask 원래부터 미적용 / **작은 box(면적<1024, score>0.1)를 seg branch anomaly 점수로 재점수화하는 처리가 있었으나 논문에 기술 없고 공개본에서 제거** / DetSeg-S 계열 클래스 여러 변형 공존
- README: 공개 checkpoint 결과는 DetSeg-R만 보고(DetSeg-S 없음), 그 표에서도 FS L&F가 두드러지게 악화 (M2A AP 78.15→72.54, FPR95 4.84→9.40)
- 상세: result_log Exp-R5

---

## 7. 재현 명령

모든 명령은 `external/DetSeg-official`에서 `conda activate detseg`, `export PYTHONPATH=.` 후 실행. `CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth`, `FS=configs/detseg/detseg-s_swin-b.py`, `RA=../../experiments/configs/detseg-s_road_anomaly.py`.

```bash
# R1: 공식 평가
python tools/test.py $FS $CK --work-dir work_dirs/repro_detseg_s/fs_lostfound

# R2 / E1 / E2 / E3: 진단 + 지표 (설정은 옵션으로)
python ../../src/eval_detseg_s_diagnostics.py --config $FS --checkpoint $CK --out-dir work_dirs/comp_eval/fs_baseline --no-save-masks
#   ① bar:        --thing-bar 0.2
#   ② size:       --small-area 1024 --small-thr 0.1
#   ③ road soft:  --road-aware soft --road-alpha 0.5
#   제안 결합:     --thing-bar 0.2 --small-area 1024 --small-thr 0.1 --road-aware soft --road-alpha 0.5
#   Road Anomaly: --config $RA

# R3: 최종 box 출처 추적 + IE-NMS 이전 후보 저장 (detection만)
python ../../src/diag_ie_nms_sources.py --config $FS --checkpoint $CK --out-dir work_dirs/diag_ie_nms/fs_lostfound

# 아래는 저장된 후보로 오프라인 실행 (repo 루트에서, GPU 불필요)
W=experiments/results/diag_ie_nms
python src/sweep_ie_nms_bars.py $W/fs_lostfound/candidates $W/road_anomaly/candidates      # E1 box 수준
python src/diag_small_object_loss.py $W/fs_lostfound/candidates --thing-bar 0.5             # R4
python src/analysis/size_thr_sim.py $W/fs_lostfound/candidates 0.2                          # E2 box 수준
python src/analysis/road_analysis.py $W/fs_lostfound/candidates external/DetSeg-official    # E3 사전 분석
python src/analysis/road_sim.py $W/fs_lostfound/candidates                                  # E3 road 규칙 비교
python src/analysis/fp_analysis.py experiments/results/diag_detseg_s/fs_lostfound/per_image.json   # R2 FP 분석
```

- `road_analysis.py`는 이미지 크기를 읽으므로 두 번째 인자로 DetSeg root를 준다 (후보 파일 안의 `img_path`가 DetSeg root 기준 상대경로)
- `vis_boxes.py`: `python src/analysis/vis_boxes.py <per_image.json> <DetSeg root> <out.jpg> 0,10,30,55`

---

## 8. 보고서 구성안 (실험 → 장 매핑)

| 장 | 내용 | 쓸 자료 |
|---|---|---|
| 1. 서론 | 도로 OoD 탐지의 필요성, 작은 장애물의 위험성, 재현성 문제 | 논문 1장, LostAndFound 동기 |
| 2. 관련 연구 | pixel uncertainty 방법(RbA, M2A), object-level(S2M, DetSeg) | notes/01~05 |
| 3. DetSeg 분석 | 구조(ID/UN query, IE-NMS, DetSeg-R/S), 코드 매핑 | notes/06, 논문 Fig.2·3 |
| 4. 재현 실험 | 환경, 데이터, 지표, R1·R2 결과 (FS 14.26 vs 32.36, RA 76.46 vs 68.35) | 6절 R1·R2 표, 오탐 그림 |
| 5. 재현 실패 원인 분석 | R3 출처 추적, R5 코드 이력, R4 작은 물체 손실 단계 | R3·R4 표, R5 커밋 표, 손실 단계 그래프 |
| 6. 개선 방법 | ① 기준값 재설정(공개본 보정), ② 크기 적응 threshold, ③ detector 자체 road 예측 기반 재점수화 | E1 스윕 표, E3 사전 분석 표 |
| 7. 실험 결과 | ablation(FS·RA), 물체 단위 지표, 정성 비교, **held-out 결과(예정)** | E3 표, 4개 막대그래프, before/after 그림(예정) |
| 8. 논의·한계 | dataset별 trade-off, 평균 IoU의 한계, 탐색 vs 검증, FS Static 버전, 작은 표본 | 10절 |
| 9. 결론 | | |

발표 핵심 메시지 (한 줄씩):
1. 공개된 ICCV 2025 코드로 작은 도로 장애물 성능이 재현되지 않았다 (14.26 vs 32.36)
2. 원인은 재학습된 가중치와 맞지 않게 남은 IE-NMS 기준값이었다 (차량 오탐 90%, 강화 0회)
3. 작은 장애물은 점수가 낮아 버려졌다 (IE-NMS 때문이 아님)
4. 재학습 없는 후처리 3가지로 IoU ×1.9, 물체 단위 F1 ×1.8, 작은 물체 검출률 ×1.7

---

## 9. 남은 작업 (우선순위)

1. **held-out 검증**: 설정을 지금 값(thing .2 / small <32² → 0.1 / road soft 0.5)으로 고정하고 SMIYC Obstacle val(30), SMIYC Anomaly val(10)에서 원본 vs 제안 1회 비교. dataset 클래스: `FSLostAndFoundDataset`에 `img_suffix='.webp'`(Obstacle) / `'.png'`(Anomaly), prefix `images`/`labels_masks`
2. **기준값 보정 절차**: 논문 방식(매칭 쌍의 최소 classification score)으로 class별 bar를 정상 주행 데이터에서 산출. 후보: Cityscapes val(가입 필요), LostAndFound train(FS L&F와 중복 확인 필요)
3. 고해상도 추론: 16px 미만 "후보 없음" 대응 (`FixScaleResize` scale 상향, 메모리 확인)
4. before/after 정성 그림 (같은 이미지에서 원본 vs 제안)
5. (선택) 공식 repo에 재현 문의 issue — 외부 게시라 사용자가 결정
6. (선택) DetSeg-R external: M2A 가중치(`~/datasets/m2a_validation_dl/Mask2Anomaly/best_contrastive.pth`)로 score map 생성

---

## 10. 주의할 점 (보고서에 반드시 반영)

- **탐색 vs 검증**: E1~E3 수치는 평가 데이터로 설정을 고른 결과다. 최종 성능은 held-out 결과로 보고하고, E1~E3는 "개발 과정 분석"으로 서술한다
- **비교 기준**: 논문 수치가 아니라 공개본 원본 실행 결과(FS 14.26, RA 76.46)다
- **size-aware의 선행 아이디어**: 저자의 연구 당시 코드에 유사한 처리(면적<1024, score>0.1, seg 점수로 재점수화)가 있었다. 우리 기여는 seg branch 없이 동작, road-aware와 결합, 진단 근거로 도출한 점으로 서술한다
- **dataset trade-off**: RA에서는 size-aware의 이득이 작고 ①+③이 더 좋다. 숨기지 않고 "작은 장애물이 많은 도시 도로 환경에서 효과가 크다"로 해석한다
- **지표 한계**: 이미지 평균 IoU는 작은 물체 개선을 반영하지 못한다. 물체 단위 지표는 SMIYC 공식 구현이 아닌 근사 구현이다
- **표본 크기**: FS 100장(큰 물체 16개), RA 60장. 큰 물체 recall 변화(75→56)는 3개 차이
- **기각된 가설도 기록**: IE-NMS의 작은 장애물 제거(6%), attention mask 주원인설(8%), 장애물형 class 보호(효과 없음), 사다리꼴 prior(정답 손실), 채점 방식 차이(전체 픽셀 IoU가 더 낮음)

---

## 11. 파일 지도

| 경로 | 내용 |
|---|---|
| `notes/06_detseg_code_analysis.md` | 코드 분석, 논문 개념 ↔ 코드 매핑 |
| `notes/07_project_direction_after_code_analysis.md` | 프로젝트 방향 |
| `notes/08_detseg_s_baseline_reproduction_plan.md` | 재현 계획, 판정 기준 |
| `notes/09_experiment_record.md` | 이 문서 |
| `experiments/result_log.md` | 실험별 원 기록 (R0~R5, E1~E3) |
| `experiments/configs/detseg-s_road_anomaly.py` | Road Anomaly override config |
| `experiments/results/` | 모든 실행의 `summary.json`, `per_image.json`, 로그, IE-NMS 이전 후보(`candidates/*.npz`), `final_boxes.json` |
| `src/eval_detseg_s_diagnostics.py` | DetSeg-S 평가·진단·물체 단위 지표·개선 옵션 |
| `src/diag_ie_nms_sources.py` | 최종 box 출처 추적, 후보 저장 |
| `src/sweep_ie_nms_bars.py` | 기준값 오프라인 스윕 |
| `src/diag_small_object_loss.py` | GT 물체 손실 단계 판정 |
| `src/analysis/` | 일회성 분석 스크립트 (FP 분석, 시각화, size/road 시뮬레이션, 라벨 확인) |
| `src/run_detseg_single_image.py` | 단일 이미지 추론 (R0) |
| `assets/qualitative_results/baseline_detseg_s/fs_lostfound_fp_examples.jpg` | 오탐 예시 그림 |
| `reports/detseg_experiment_log.html` | 시각화 정리 페이지 (claude.ai artifact와 동일 내용) |
| `~/datasets/` (repo 밖) | 데이터 원본. `external/DetSeg-official/data/`에 symlink |
| `external/DetSeg-official/work_dirs/` (gitignore) | 실행 원본 출력 (예측 mask PNG 포함) |
