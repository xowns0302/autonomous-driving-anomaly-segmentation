# 01. 논문 핵심 정리: DetSeg

> Zhu, Liu, Sun, Zou, Xia, Xu. **Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes.** ICCV 2025, pp. 8472–8481.
> 코드: https://github.com/huachao0124/DetSeg-official
>
> 이 문서는 논문 원문만을 근거로 정리한다. 공개 코드와 다른 부분은 각 절 끝의 "코드 확인" 줄에 따로 적는다.

---

## 1. 한 줄 요약

**픽셀마다 "이상한 정도"를 점수로 매기는 대신, 도로 위 모든 물체를 box로 찾은 뒤 "아는 물체(ID)"를 지우고 남은 box를 이상 물체(OoD)로 쓴다.** (detecting-all-then-suppressing)

---

## 2. 문제의식 (1장, Figure 1)

기존 방법(RbA, Mask2Anomaly 등)은 Cityscapes로 학습한 segmentation 모델의 **픽셀별 불확실성**을 anomaly score로 쓴다. 문제는 세 가지다.

| 문제 | 내용 |
|---|---|
| 경계 오탐 | segmentation 모델은 물체 경계 같은 애매한 곳에서 원래 불확실하다 → 최적 threshold에서도 경계 오탐이 남는다 |
| threshold 선택 | 낮으면 오탐이 흩어지고(scattered FP), 높으면 mask가 끊긴다(fragmented) |
| 프레임 간 불일치 | 연속 프레임에서 최적 threshold가 흔들려 주행 계획을 혼란시킬 수 있다 |

핵심 진단: **기존 방법에는 "물체 단위 이해(object-level understanding)"가 없다.**

---

## 3. 방법

### 3.1 OoD box 생성 (3.1절, Figure 2)

**기반 모델: Grounding DINO** (텍스트로 찾을 대상을 지정하는 open-vocabulary detector)
- 사전학습 가중치를 그대로 쓴다. 도로 데이터로 재학습하지 않는다
- ID class를 텍스트로 정할 수 있어 유연하다 (예: 라바콘을 상황에 따라 정상/이상으로 지정 가능)

**텍스트 입력:** 학습 가능한 universal token(**U-T**) + Cityscapes 19개 class 이름

**두 종류의 query (dual-query)**

| | ID query | UN (universal) query |
|---|---|---|
| 역할 | 19개 ID class 물체를 찾고 class를 맞힌다 | class와 무관하게 **모든 물체**를 찾는다 |
| 점수 | classification score (텍스트 유사도) | **objectness score** (새로 추가한 objectness head) |
| 출발점 | Grounding DINO의 Language-guided Query Selection | 새로 추가 (300개) |

**정보 격리 (attention mask)**
- ID query ↔ UN query 사이 self-attention을 막는다
- query-text cross-attention에서 **ID query는 U-T를 못 보고, UN query는 ID class 이름을 못 본다**
- 목적: ID 검출 능력은 그대로 두고, UN query는 ID·OoD 물체를 모두 보게 한다

**IE-NMS (In-distribution Enhanced NMS)** (Figure 3)
1. **보정 단계 (statistical calibration):** Cityscapes 일부에서 class마다 예측 box와 정답 box를 1:1 매칭하고, IoU < 0.5 쌍은 버린 뒤, **남은 쌍의 최소 classification score를 그 class의 bar로 기록**한다 (예시: car 0.36, truck 0.58, bicycle 0.28)
2. bar를 넘은 ID box의 score를 **λ배** 올린다 → "확실히 아는 물체"
3. NMS 수행 → 강화된 ID box가 같은 자리의 UN box를 지운다
4. NMS 후 **ID box는 모두 버린다** → OoD 후보만 남음
5. **objectness score > τ = 0.2**인 box만 최종 결과로 남긴다

**τ = 0.2 고정에 대한 논문 주장:** 픽셀 점수 지도는 threshold에 민감하지만, 물체 단위 접근은 threshold 안정성이 훨씬 높아 **표준 값 하나로 일관된 성능**을 낸다. → DetSeg의 핵심 주장 중 하나

> 코드 확인: λ = 2. bar는 보정값이 아니라 `[0.2]*11 + [0.5]*8`(stuff/thing)로 하드코딩되어 있다. 보정 절차 코드는 없다. self-attention mask는 적용되지만, query-text mask(`query_text_mask`)는 생성만 되고 decoder에 전달되지 않는다.

### 3.2 학습 (3.2절, 4.1절)

- **대부분의 파라미터를 고정**하고 U-T, UN query, box 회귀 head F_b, objectness head F_o만 fine-tuning
- 데이터: **COCO**, 1 epoch, AdamW, lr 1e-4, A100 2장
- 손실 (식 2): Hungarian matching 후 매칭된 query는 L1 + GIoU + Focal(s, 1), 매칭 안 된 query는 Focal(s, 0)
- UN query 수 300, 나머지 설정은 Grounding DINO와 동일

> 코드 확인: 공개 checkpoint(`detseg_swin-b_coco_20260623`)가 이 학습의 결과물이다. 단 README에 "논문 이후 같은 config로 **재학습**한 것"이라고 명시되어 있다.

### 3.3 두 가지 활용 방식

| | **DetSeg-R** (Refinement) | **DetSeg-S** (Segmentation) |
|---|---|---|
| 입력 | 기존 방법의 **픽셀 anomaly score map** + DetSeg OoD box | 이미지 + DetSeg OoD box |
| 동작 | **OoD box 밖의 anomaly score를 억제**해 흩어진 오탐 제거 | box를 **SAM**의 prompt로 넣어 binary mask 생성 |
| 출력 | 개선된 score map (연속값) | binary anomaly mask |
| threshold 탐색 | 여전히 필요 (score map이므로) | **불필요** |
| 평가 지표 | AUROC, AP, FPR95 (+ 최적 threshold IoU) | IoU |
| 혼자 동작? | ✗ (기존 방법이 있어야 함) | ✓ |

> 코드 확인: DetSeg-R은 "box 밖 억제" 대신 "box 안 score + 1.0"으로 구현되어 있다 (순위 기준 지표에서는 사실상 같은 효과). SAM은 `facebook/sam-vit-base`. 논문은 SAM 크기를 명시하지 않는다.

### 3.4 VPHM (3.3절, Figure 4) — 비디오 시간 smoothing

- 학습 없는 알고리즘. 물체는 소실점(VP)에서 자차 방향으로 방사형으로 이동한다는 원근 prior를 쓴다
- 이전 프레임 물체마다 VP에서 box 좌우 중점을 지나는 **부채꼴 영역**을 만들고, 현재 프레임 물체 위치에 pseudo box를 만들어 **IoU를 Hungarian matching 비용**으로 쓴다
- IoU < 0.5 매칭은 버리고, 매칭된 물체의 objectness를 **momentum smoothing** (식 3, w = 0.2)
- 평가는 LostAndFound 비디오(Table 4)와 Figure 6에서만

> 코드 확인: 실행 가능한 구현이 없다 (`VanishingPointGuidedCost`만 있고 미완성, 호출처 없음).

---

## 4. 실험 설정 (4.1절)

### 4.1 데이터셋

| Dataset | 규모 | 특징 | 사용 표 |
|---|---|---|---|
| Road Anomaly | 60장 | 인터넷 이미지, 동물·돌·라바콘 | T1, T2, T5, T6 |
| FS L&F | 100장 | LostAndFound에서 뽑은 정밀 라벨, 작은 장애물 | T1, T2, T5, T6 |
| FS Static | (val) | Cityscapes val에 Pascal VOC 물체 합성 | T1, T2, T5, T6 |
| SMIYC Anomaly | 110장 (test) | 다양한 물체 | T2 (val), T3 (test 서버) |
| SMIYC Obstacle | 442장 (test) | 작은 물체, 다양한 날씨 | T2 (val), T3 (test 서버) |
| LostAndFound | 112 시퀀스, 2,104 프레임 | 비디오 | T4 (test) |

- **FS Static 주의:** 기존 방법들(PEBAL, RPL, M2A, S2M, RbA)이 쓴 FS Static val set에 불일치가 있어, 저자들이 **일관된 버전에서 기존 방법을 공식 구현·가중치로 다시 돌렸다**

### 4.2 평가 지표

| 지표 | 대상 | 정의 |
|---|---|---|
| AUROC, AP, FPR95 | score map (기존 방법, DetSeg-R) | 픽셀 단위. RA, Fishyscapes, LostAndFound |
| IoU | binary mask | RA, Fishyscapes, SMIYC val. **score map 방법은 0.01 간격으로 threshold를 탐색해 IoU가 최대인 값을 고른다** (oracle) |
| sIoU, PPV, mean F1 | 물체 단위 | SMIYC 공식 서버 (test) |

### 4.3 비교 대상 (기존 방법)

| 방법 | 참고문헌 | 표 |
|---|---|---|
| PEBAL | [37] | T1, T2, T4 |
| RPL+CoroCL | [30] | T1, T2, T4 |
| Mask2Anomaly (M2A) | [34] | T1, T2, T3, T4 |
| RbA, RbA OoD | [32] | T1, T2, T3, T4, T5 |
| UNO | [10] | T1, T2, T3, T4 |
| S2M | [40] | T2 (threshold 탐색 없는 유일한 비교 대상) |
| 그 외 SynthCP, SML, Meta-OoD, SynBoost, DenseHybrid 등 | | T1, T2 (인용 수치) |

---

## 5. 주요 결과

### 5.1 Table 1 — DetSeg-R의 픽셀 지표 개선 (발췌)

| 방법 | RA AP | RA FPR95 | FS L&F AP | FS L&F FPR95 | FS Static AP | FS Static FPR95 |
|---|---|---|---|---|---|---|
| Mask2Anomaly | 79.70 | 13.45 | 69.46 | 9.31 | 82.37 | 7.20 |
| + DetSeg-R | **87.57** | **5.7** | **78.15** | **4.84** | **93.12** | **1.71** |
| RPL+CoroCL | 71.36 | 18.43 | 70.62 | 2.52 | 92.47 | 0.85 |
| + DetSeg-R | **79.58** | **10.16** | **75.23** | **2.15** | **93.66** | **0.63** |
| RbA | 78.45 | 11.83 | 60.96 | 10.63 | 62.86 | 41.83 |
| + DetSeg-R | **86.12** | **11.12** | **69.39** | **6.53** | **80.89** | **4.38** |

- 모든 기존 방법에서 개선. 최대 FPR95 감소 37.45%p (RbA, FS Static)
- 예외: UNO + DetSeg-R은 FS L&F FPR95가 1.30 → 1.95로 악화

### 5.2 Table 2 — binary mask IoU

| 방법 | threshold 탐색 | RA | FS L&F | FS Static | SMIYC Anomaly | SMIYC Obstacle |
|---|---|---|---|---|---|---|
| RPL+CoroCL | ✓ | 50.97 | 15.75 | 36.46 | 68.80 | 28.66 |
| Mask2Anomaly | ✓ | 58.43 | 34.35 | 60.51 | 83.74 | 67.13 |
| + DetSeg-R | ✓ | 65.27 | 35.46 | 75.61 | 82.51 | 72.83 |
| RbA | ✓ | 51.33 | 22.42 | 39.28 | 78.08 | 63.67 |
| UNO | ✓ | 69.33 | 50.22 | 85.81 | 77.43 | 76.04 |
| + DetSeg-R | ✓ | 74.66 | 50.78 | 86.52 | 78.05 | 78.73 |
| S2M | × | 58.39 | 30.46 | 69.99 | 77.54 | 67.64 |
| **DetSeg-S** | **×** | **68.35** | **32.36** | **83.34** | 75.82 | **81.58** |

- DetSeg-S는 threshold 탐색 없이 S2M보다 RA +9.96, FS L&F +1.90, FS Static +13.35
- 주의: FS L&F에서 DetSeg-S(32.36)는 **threshold 탐색을 한 M2A(34.35)·UNO(50.22)보다 낮다**

### 5.3 Table 3 — SMIYC test (공식 서버)

UNO + DetSeg-R: Anomaly mean F1 62.61 → 73.09, Obstacle mean F1 77.65 → 80.84. Anomaly track의 AP·FPR95는 악화. 원인 설명: box 안의 물체 아닌 영역이 새 오탐이 될 수 있다.

### 5.4 Table 4 — LostAndFound test (비디오)

DetSeg-R이 RPL, M2A, RbA OoD, UNO 모두 개선. VPHM 추가 시 대부분 추가 개선 (예: M2A AP 82.95 → 85.78).

### 5.5 Ablation (4.3절, Table 5·6)

- **baseline:** Grounding DINO에 prompt "objects"만 준 것
- Table 5(DetSeg-R): RbA score map 사용, backbone 공유
- Table 6(DetSeg-S IoU):

| UN query | Attention mask | IE-NMS | RA | FS L&F | FS Static |
|---|---|---|---|---|---|
| – | – | – | 61.21 | 19.39 | 38.46 |
| ✓ | – | – | 44.10 | **2.00** | 16.84 |
| ✓ | ✓ | – | 60.89 | 7.29 | 36.60 |
| ✓ | ✓ | ✓ | **68.35** | **32.36** | **83.34** |

- UN query만 넣으면 오히려 나빠진다 (attention mask 없으면 UN query가 ID 물체 쪽으로 수렴)
- **IE-NMS가 FS L&F·FS Static에서 결정적** (7.29 → 32.36, 36.60 → 83.34)
- 지연시간: RbA 148ms → DetSeg-R 225ms (+77ms)

---

## 6. 논문의 기여 (저자 주장)

1. 물체 단위 이해로 비물체 영역 오탐을 줄이고, threshold 탐색 없이 anomaly mask를 만드는 프레임워크
2. 소실점 prior로 프레임 간 예측을 smoothing하는 VPHM
3. 여러 벤치마크에서 SOTA 대비 개선

## 7. 논문이 밝힌 한계 (5장)

- 효율: DetSeg-R이 이미지당 약 77ms 추가
- VPHM은 정적 물체 위주. 움직이는 물체는 시간 일관성을 다 잡지 못할 수 있음

## 8. 우리 프로젝트 관점에서 중요한 포인트

| 포인트 | 왜 중요한가 |
|---|---|
| τ = 0.2 고정이 핵심 주장 | 개선할 때 최종 threshold를 바꾸면 논문 철학과 충돌한다. 바꾸지 않는 방향을 우선한다 |
| IE-NMS bar는 **데이터로 보정**하는 값 | 공개 코드에는 보정 절차가 없다 → 논문대로 복원할 수 있는 부분 |
| IE-NMS가 FS L&F에서 결정적 (Table 6) | 재현 여부를 가르는 핵심 모듈 |
| DetSeg-R은 기존 방법이 있어야 평가 가능 | 기존 방법 score map 확보가 DetSeg-R 실험의 전제 |
| Table 2 IoU는 기존 방법에 oracle threshold를 준 값 | 공정 비교 아님을 논문도 인정 (DetSeg-S는 탐색 없음) |
| FS L&F에서 DetSeg-S < M2A(탐색) | 작은 장애물이 DetSeg-S의 약점 → 개선 여지 |
| SMIYC test·LostAndFound는 서버 제출·대용량 | 범위 결정 시 고려 |
