# F1. 정성 결과 그림과 이미지별 IoU

- 날짜: 2026-10-18
- 목적: 보고서·발표용 그림. 숫자(평균 IoU)가 실제로 어떤 장면 차이에서 나왔는지 보여주고, 실패 사례도 함께 보여준다
- 결론: 저장한 mask로 다시 계산한 평균 IoU가 이전 결과와 **모든 자리 일치**. C5의 이득은 "IE-NMS가 이상 물체를 지우던 장면"에서 나오고, C7(C8)의 손해는 "작은 이상 물체(콘 등)를 아는 물체로 보고 지우는 장면"에서 나온다

## 1. 방법

1. `bash scripts/save_masks_for_figures.sh`: DetSeg-S의 4가지 방법(공개본, 논문 방식 bar, C5, C8)으로 4개 데이터셋을 다시 실행하고, 이미지마다 예측 mask를 `results/F1/masks/<데이터셋>/<방법>/<이름>.png`에 저장한다 (`run_with_bars.py --save-masks`)
2. `python src/make_qualitative.py`: 이미지별 IoU를 계산하고(`results/F1/per_image_iou.csv`), **정해진 규칙으로** 이미지를 고른다 (직접 고르지 않음)
   - C5 이득 상위 2장: C5 − 논문 방식 bar 의 IoU 차이가 가장 큰 이미지 (1점 넘게 좋아진 것만)
   - C8 손해 1장: C8 − C5 가 가장 작은 이미지 (1점 넘게 나빠진 것만, 위와 겹치면 제외)

그림 읽는 법: 열 = GT | 공개본 | 논문 방식 bar (baseline) | C5 | C8. 빨강 = 예측, 초록 선 = GT 이상 물체, **어둡게 칠한 곳 = 평가에서 무시하는 영역(void)**. 이 영역의 빨강은 IoU에 영향이 없다 (FS L&F·SMIYC Obstacle은 도로 밖이 void).

## 2. 확인: 저장한 mask로 다시 잰 평균 IoU

| 데이터셋 | 공개본 | 논문 방식 bar | C5 | C8 |
|---|---|---|---|---|
| FS L&F (100장) | 14.26 | 24.26 | 32.67 | 38.44 |
| RA (60장) | 76.46 | 79.25 | 76.82 | 76.39 |
| SMIYC Obstacle (30장) | 77.93 | 77.93 | 80.93 | 81.03 |
| SMIYC Anomaly (10장) | 69.12 | 54.57 | 78.06 | 69.58 |

E1, A3, I4, I6의 공식 평가 수치와 같다.

## 3. 이미지별로 몇 장이 바뀌었나 (1 IoU점 넘는 변화만 셈)

| 데이터셋 | C5 > baseline | C5 < baseline | C8 > C5 | C8 < C5 |
|---|---|---|---|---|
| FS L&F | 11 | 4 | 29 | 1 |
| RA | 4 | **18** | 7 | 5 |
| SMIYC Obstacle | 1 | 0 | 1 | 0 |
| SMIYC Anomaly | 3 | 1 | 0 | 1 |

- SMIYC Obstacle의 +3.0은 **이미지 1장**(25번)에서 나온 것이다. 검증셋이 작다는 한계(09 §6)의 구체적 근거
- RA에서 C5는 18장에서 나빠지고 4장에서 좋아졌다 (평균 −2.43). C5가 보호한 box 중 일부가 실제로는 오탐이었다는 뜻
- C8의 C7 단계는 FS L&F에서 29장을 좋게 하고 1장만 나쁘게 했다 (오탐 제거)

## 4. 그림과 관찰

### FS L&F — `assets/figures/qual_fs_lostfound.jpg`

- **65, 68 (C5 이득):** baseline(논문 방식 bar)은 도로 위 작은 장난감 차를 "car"로 보고 IE-NMS에서 지워 IoU 0. C5는 M2A 점수가 높아 보호 → 92.7 / 92.3. 공개본은 물체는 찾지만 뒤쪽 주차 차량을 오탐으로 함께 잡아 23.8 / 56.1
- **18 (C8 손해):** 멀리 있는 아주 작은 물체들. C5까지는 일부를 잡지만(3.0) C7이 지워 0.0. 효과가 큰 사례는 아니지만 C7이 작은 물체에 불리하다는 경향을 보여준다

### Road Anomaly — `assets/figures/qual_road_anomaly.jpg`

- **vehicle02 (C5 이득):** 스키드로더(공사 차량)를 baseline은 "truck"과 겹친다고 지움(12.1). C5가 보호 → 90.4. 논문이 말한 "아는 클래스와 비슷한 이상 물체" 문제가 실제로 고쳐지는 장면
- **obstacles12_rocks2 (C5 이득):** 도로 위 돌 여러 개. baseline 3.3 → C5 71.0
- **cones05 (C5·C8 손해):** baseline 66.2 → C5 35.7 → C8 14.3. C5가 길가 덤불·주차 차량 box를 보호해서 오탐이 늘고, C7은 콘을 "아는 물체(prototype과 비슷함)"로 보고 지운다. RA에서 C5가 평균 −2.43, C8이 추가로 −0.43인 이유를 보여주는 대표 사례

### SMIYC Obstacle — `assets/figures/qual_smiyc_obstacle.jpg`

- **25 (C5 이득):** 도로 위 작은 검은 물체. 공개본·baseline 모두 0.0 → C5·C8 90.0. 이 데이터셋에서 1점 넘게 바뀐 이미지는 이 한 장뿐 (C8 손해 사례 없음)

### SMIYC Anomaly — `assets/figures/qual_smiyc_anomaly.jpg`

- **9 (C5 이득):** 트랙터 + 마차. baseline은 "car/truck" box로 지움(3.4) → C5 94.0
- **5 (C5 이득, 그리고 C8 손해):** 말 트레일러 전체가 GT. 공개본·baseline·C8은 안의 말만 잡아 약 12. C5는 트레일러 전체 box를 보호해 97.3. C8에서는 C7이 트레일러 box를 "아는 물체"로 보고 지워 11.9로 돌아간다. SMIYC Anomaly에서 C8(69.58)이 C5(78.06)보다 낮은 이유의 대부분이 이 한 장이다 (10장 평균에서 −8.5 ≈ 이 이미지 −85.4 / 10)

## 5. 결과 파일

- 그림: `assets/figures/qual_{fs_lostfound,road_anomaly,smiyc_obstacle,smiyc_anomaly}.jpg`
- 이미지별 IoU: `results/F1/per_image_iou.csv` (열: dataset, image, released, paper, c5, c8)
- mask: `results/F1/masks/` (스크립트로 다시 만들 수 있음)
