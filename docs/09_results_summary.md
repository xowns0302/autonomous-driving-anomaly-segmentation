# 09. 결과 종합 (보고서용)

- 작성: 2026-10-18
- 이 문서는 모든 실험 결과를 한곳에 모은다. 각 수치의 근거는 괄호 안 실험 기록에 있다 ([docs/experiments/](experiments/))
- 환경: RTX 4070 Ti, WSL2, PyTorch 2.1.0 + CUDA 11.8 ([03](03_environment_and_data.md))

---

## 1. 한 문단 요약

공개된 DetSeg(ICCV 2025) 코드와 가중치를 수정 없이 실행하면 Road Anomaly와 SMIYC는 논문과 비슷하지만, FS Lost & Found에서는 DetSeg-S IoU가 14.26으로 논문(32.36)의 절반에 못 미쳤다. 같은 환경에서 기존 방법 Mask2Anomaly는 논문 수치를 소수 둘째 자리까지 재현해, 차이가 데이터나 평가 방식이 아니라 DetSeg 공개본에 있음을 확인했다. 원인은 논문에 기술된 IE-NMS 기준값(bar) 보정 절차가 공개 코드에서 빠지고, 재학습 전 가중치 기준의 값(0.5)이 남은 것이었다. 논문 방식으로 bar를 Cityscapes에서 보정하자(약 0.1) FS L&F가 24.26으로 회복됐지만, 진짜 이상 물체가 ID 물체로 오인되어 지워지는 과잉 억제가 늘었다(SMIYC Anomaly 69.12 → 54.57). 이를 해결하기 위해 픽셀 분류기 Mask2Anomaly로 IE-NMS의 억제를 확인하는 규칙(C5)을 제안했고, 튜닝셋에서만 설정을 고른 뒤 검증셋에서 SMIYC Obstacle 77.93 → 80.93, SMIYC Anomaly 54.57 → 78.06으로 개선했다. 추가로 Finding Dino(PROWL)의 DINOv2 prototype 신호로 남은 오탐을 제거하는 규칙(C7)을 결합하면(C8) 도시 도로 장애물(FS L&F)에서 38.44까지 오르지만, 처리 시간이 늘고 ID 물체를 닮은 이상 물체를 놓치는 trade-off가 있다. 처리 시간 분석에서는 DetSeg-S 시간의 절반이 SAM의 CPU 전·후처리였고, 결과를 바꾸지 않는 최적화로 이를 줄였다.

---

## 2. 재현 (Phase 1)

### 2.1 DetSeg-S (논문 Table 2, IoU) — [E1](experiments/E1_detseg_s_reproduction.md)

| Dataset | 논문 | 공개본 (우리) | 판정 |
|---|---|---|---|
| Road Anomaly | 68.35 | 76.46 | 부분 재현 (더 높음) |
| FS L&F | 32.36 | **14.26** | **불일치** |
| FS Static | 83.34 | 33.51 | 비교 불가 (데이터 버전 차이: 우리 버전은 30장 중 10장에 OoD가 없어 최대 66.7) |
| SMIYC Anomaly val | 75.82 | 69.12 | 부분 재현 |
| SMIYC Obstacle val | 81.58 | 77.93 | 부분 재현 |

### 2.2 기존 방법 Mask2Anomaly 단독 (논문 Table 1·2) — [E2](experiments/E2_E3_m2a_and_detseg_r.md)

| Dataset | AP 논문 / 우리 | FPR95 논문 / 우리 | IoU(최적 threshold) 논문 / 우리 |
|---|---|---|---|
| FS L&F | 69.46 / **69.44** | 9.31 / **9.31** | 34.35 / **34.33** |
| Road Anomaly | 79.70 / 82.05 | 13.45 / 15.73 | 58.43 / **58.41** |
| SMIYC Anomaly | – | – | 83.74 / 84.37 |
| SMIYC Obstacle | – | – | 67.13 / 67.17 |

→ 우리 데이터와 평가 방식은 논문과 같다.

### 2.3 Mask2Anomaly + DetSeg-R — [E3](experiments/E2_E3_m2a_and_detseg_r.md)

| Dataset | AP 개선폭 논문 / 우리 | IoU 개선폭 논문 / 우리 |
|---|---|---|
| Road Anomaly | +7.87 / +6.22 | +6.84 / +11.77 |
| SMIYC Obstacle | – | +5.70 / +5.25 |
| FS L&F | +8.69 / **+1.40** | +1.11 / **−3.18** |
| SMIYC Anomaly | – | −1.23 / **−13.22** |

→ DetSeg-R도 DetSeg-S와 같은 데이터셋에서 실패 (같은 OoD box를 쓰므로)

### 2.4 Ablation (논문 Table 6, DetSeg-S IoU) — [E4](experiments/E4_detseg_s_ablation.md)

| | RA | FS L&F | SMIYC A | SMIYC O |
|---|---|---|---|---|
| Grounding DINO + "objects" | 53.82 | 2.72 | 85.62 | 39.15 |
| + UN query (IE-NMS 없음) | 69.44 | 4.99 | 75.06 | 77.93 |
| + IE-NMS (전체) | 76.46 | 14.26 | 69.12 | 77.93 |

→ 논문과 같은 경향 (UN query, IE-NMS 순으로 향상). IE-NMS 효과가 큼

---

## 3. 원인 분석 (Phase 2)

| 분석 | 발견 | 근거 |
|---|---|---|
| 공식 저장소 이력 | 공개 코드는 논문 약 9개월 뒤 재작성, 가중치 재학습. bar 보정 절차 코드 없음, 값(0.2 / 0.5)만 남음 | git 이력, README ([06](06_detseg_code_analysis.md)) |
| [A2](experiments/A2_failure_diagnosis.md) 과소 억제 | FS L&F 오탐 후보의 41%가 IE-NMS 뒤 남음. 남은 오탐의 대부분이 차·사람 위이며, 그 ID box 점수(중앙 0.26)가 bar 0.5를 넘지 못함 | IE-NMS 전후 추적 |
| A2 과잉 억제 | SMIYC Anomaly에서 진짜 이상 물체가 "truck"(0.57) box, "road" box에 지워짐 | 이미지 4, 5 |
| A2 작은 물체 | FS L&F 16px 미만 정답의 45%는 후보조차 없고 42%는 objectness < 0.2. IE-NMS 손실은 2% | 손실 단계 판정 |
| [A3](experiments/A3_ie_nms_bar_calibration.md) bar 보정 | 논문 방식 bar ≈ 0.085~0.143 (공개 0.5) | Cityscapes val 500장 |

### bar 보정 효과 (DetSeg-S IoU)

| | FS L&F | RA | SMIYC O | SMIYC A |
|---|---|---|---|---|
| 공개본 | 14.26 | 76.46 | 77.93 | 69.12 |
| 논문 방식 bar (이후 기준선) | **24.26** | 79.25 | 77.93 | **54.57** |

→ FS L&F는 "불일치 → 부분 재현"으로 회복. 그러나 과잉 억제로 SMIYC Anomaly 하락. **bar 하나로는 두 실패를 함께 고칠 수 없다**

---

## 4. 개선 (Phase 3)

규칙: 재학습 없음, τ = 0.2 고정, 하이퍼파라미터는 튜닝셋(FS L&F + RA)에서만, 검증셋(SMIYC Obstacle + Anomaly)은 설정 고정 후 1회 ([05](05_improvement_plan.md))

### 4.1 시도한 후보와 결과

| ID | 아이디어 | 결과 | 핵심 이유 |
|---|---|---|---|
| C1 | 배경 class box는 OoD를 못 지움 | ❌ 기각 ([I1](experiments/I1_suppression_rules.md)) | 배경 box는 거대한 배경 오탐 box를 지우는 역할을 하고 있었음 (새로 살아난 box 100% 오탐) |
| C2 | objectness 높으면 보호 | ❌ 기각 (I1) | objectness는 COCO로 학습되어 **차·사람에서 가장 높음** (새로 살아난 오탐 중앙 0.88) |
| C3 | 도로 밖 box 점수 낮춤 | ❌ 효과 없음 ([I2](experiments/I2_road_context.md)) | bar 보정이 도로 밖 오탐을 이미 제거 (297 → 38개) |
| C4 | 고해상도 입력 | ❌ 기각 ([I3](experiments/I3_high_resolution.md)) | 학습 해상도 밖에서 box 예측 붕괴 (1.5배: FS 0.32) |
| **C5** | **M2A 점수 ≥ 0.5인 OoD box는 ID box가 못 지움** | ✅ **성공** ([I4](experiments/I4_m2a_verified_ie_nms.md)) | M2A는 차·사람에 낮은 점수 → 진짜 이상 물체만 보호 |
| C6 | PROWL 신호로 보호 | ❌ 기각 ([I5](experiments/I5_prowl_rules.md)) | RA의 ID 물체가 Cityscapes prototype과 닮지 않음 |
| C7 | PROWL로 "아는 물체와 닮은" OoD 제거 | △ 튜닝셋만 효과 (I5) | 검증셋의 문제는 오탐이 아니라 과잉 억제 |
| **C8** | **C5 + C7** | ✅ 성공, trade-off ([I6](experiments/I6_c5_c7_combined.md)) | 도시 도로 장애물 최고, ID를 닮은 큰 이상 물체는 놓침 |

### 4.2 최종 비교: DetSeg-S IoU

| 방법 | FS L&F (튜닝) | RA (튜닝) | **SMIYC Obstacle** (검증) | **SMIYC Anomaly** (검증) |
|---|---|---|---|---|
| 논문 Table 2 | 32.36 | 68.35 | 81.58 | 75.82 |
| 공개본 | 14.26 | 76.46 | 77.93 | 69.12 |
| 기준선: 논문 방식 bar | 24.26 | 79.25 | 77.93 | 54.57 |
| **C5 (제안)** | 32.67 | 76.82 | 80.93 | **78.06** |
| **C8 (대안)** | **38.44** | 76.39 | **81.03** | 69.58 |
| (참고) M2A 단독, 최적 threshold | 34.33 | 58.41 | 67.17 | 84.37 |

### 4.3 최종 비교: DetSeg-R (Mask2Anomaly + DetSeg-R), AP / FPR95 / 최적 IoU

| Dataset | M2A 단독 | 공개본 DetSeg-R | 기준선 | **C5** | **C8** |
|---|---|---|---|---|---|
| FS L&F | 69.44 / 9.31 / 34.33 | 70.84 / 9.39 / 31.15 | 73.19 / 8.09 / 29.15 | 77.59 / 6.64 / 36.43 | **83.15 / 6.17 / 37.71** |
| RA | 82.05 / 15.73 / 58.41 | **88.27** / 8.92 / **70.18** | 87.97 / 9.40 / 68.77 | 87.32 / **8.87** / 67.03 | 87.13 / 9.28 / 66.05 |
| SMIYC O | 92.83 / 0.16 / 67.17 | 97.86 / 0.04 / 72.42 | 97.86 / 0.04 / 72.42 | **98.24 / 0.03** / 75.44 | **98.24 / 0.03 / 75.50** |
| SMIYC A | **95.49 / 2.39 / 84.37** | 81.98 / 12.71 / 71.15 | 69.56 / 10.44 / 58.29 | 91.63 / 12.60 / 79.17 | 82.10 / 12.59 / 71.13 |

---

## 5. 처리 시간

### 5.1 baseline ([T2](experiments/T2_latency_baseline.md), ms/이미지, 30장 × 3회)

| 방법 | FS L&F | RA | SMIYC O | SMIYC A |
|---|---|---|---|---|
| M2A 단독 | 498 | 405 | 483 | 394 |
| DetSeg-S (공개본) | 808 | 679 | 655 | 654 |
| DetSeg-S (논문 방식 bar) | 603 | 615 | 659 | 608 |
| DetSeg-S + C5 | 1,189 | 1,068 | 1,192 | 1,043 |
| DetSeg-S + C8 | 1,336 | 1,178 | 1,373 | 1,148 |
| M2A + DetSeg-R + C5 | 857 | 691 | 853 | 667 |

- DetSeg-S 시간의 약 절반이 SAM이고, 그중 대부분이 CPU 전·후처리
- DetSeg-R의 DetSeg 부분은 227 ms로 논문 Table 5(225 ms)와 같다

### 5.2 결과를 바꾸지 않는 최적화 후 ([T3](experiments/T3_speedups.md))

SAM 후처리(쓰는 mask 1장만, GPU), SAM 전처리(이미지 1회), BERT 재사용, C5·C7 계산 정리. 12개 실행에서 모든 정확도 지표가 소수 둘째 자리까지 같음을 확인.

| 방법 | FS L&F | RA | SMIYC O | SMIYC A |
|---|---|---|---|---|
| DetSeg-S (공개본) | 808 → **532** (−34%) | 679 → **433** (−36%) | 655 → **545** (−17%) | 654 → **414** (−37%) |
| DetSeg-S + C5 | 1,189 → **1,040** (−13%) | 1,068 → 852 (−20%) | 1,192 → 1,068 (−10%) | 1,043 → 796 (−24%) |
| DetSeg-S + C8 | 1,336 → **1,106** (−17%) | 1,178 → 901 (−23%) | 1,373 → 1,134 (−17%) | 1,148 → 852 (−26%) |
| M2A + DetSeg-R + C5 | 857 → 817 (−5%) | 691 → 660 (−5%) | 853 → 819 (−4%) | 667 → 635 (−5%) |

- 가장 큰 절감은 SAM (437 → 227 ms, FS L&F). BERT 28 → 0.3 ms
- 남은 C5·C8 비용의 대부분은 Mask2Anomaly (약 400~500 ms)

---

## 6. 한계

1. FS Static은 데이터 버전 불일치로 비교하지 못했다
2. 검증셋이 작다 (SMIYC Obstacle 30장, Anomaly 10장). SMIYC Anomaly의 큰 차이(+23.5)는 소수 이미지(A2의 이미지 4, 5 등)의 영향이 크다
3. C5·C6·C7의 아이디어는 검증셋 실패 사례(SMIYC Anomaly)를 본 뒤 떠올렸다. 숫자는 튜닝셋에서만 골랐다
4. C5·C8은 Mask2Anomaly를 함께 써서 "DetSeg-S 단독"이 아니며 처리 시간이 늘어난다 (DetSeg-R은 원래 M2A를 사용)
5. 튜닝셋이 검증셋의 이상 물체 분포를 대표하지 못한 경우가 있었다 (C8: 튜닝셋에서 C5보다 좋았지만 SMIYC Anomaly에서 역전)
6. 논문 원래 가중치가 공개되지 않아 "원래 가중치 + 원래 bar"의 수치는 확인할 수 없다
7. VPHM(비디오), SMIYC test 서버, DetSeg-R 내부 seg branch는 범위에서 제외했다
