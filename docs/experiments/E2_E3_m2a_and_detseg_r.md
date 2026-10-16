# E2·E3. Mask2Anomaly 단독 vs + DetSeg-R (논문 Table 1·2)

- 날짜: 2026-10-16
- 단계: Phase 1 (재현). 공개 코드·가중치 그대로

## 1. 목적

- **E2:** 기존 방법(Mask2Anomaly) 단독 성능이 우리 환경에서 논문 수치와 같은가? → 우리 데이터·평가 파이프라인이 논문과 같은지 검증하는 기준점
- **E3:** 같은 score map에 DetSeg-R을 적용하면 논문처럼 개선되는가?

## 2. 설정

| 항목 | 값 |
|---|---|
| M2A 코드 | `external/Mask2Anomaly` (공식, `5c84283`), conda `m2a` |
| M2A 가중치 | `best_contrastive.pth` (공식 Google Drive) |
| M2A score | 공식 `anomaly_inference.py`와 같은 계산: `1 − max softmax` + anomaly mask 보정, 원본·좌우반전 평균 (`src/m2a_save_scores.py`) |
| score map | `results/score_maps/m2a/{dataset}/{image}.npy` (git 제외, 1.5GB) |
| DetSeg-R | 공식 `DetSegR`, `detseg-r_swin-b_external.py` 기반 (`configs/detseg-r_m2a_{dataset}.py`): box_score_thr 0.2, box 내부 +1.0, road filter·SAM refine 끔 (논문 재현 기본값) |
| E2 실행 방식 | 같은 config에 `model.box_score_thr=1.1` → box가 하나도 남지 않아 score map이 그대로 평가됨 (E2와 E3가 같은 파이프라인·지표를 공유) |
| 지표 | `AnomalyMetricLoad` (AUROC, AP, FPR95: 모든 이미지의 픽셀을 합쳐 계산), `OracleThresholdAnomalyIoUMetric` (0.01 간격 threshold 탐색으로 최대 IoU, 논문 4.1절과 같은 정의) |

## 3. 실행 명령

```bash
# score map (m2a env, repo 루트)
python src/m2a_save_scores.py --m2a-repo external/Mask2Anomaly \
  --weights ~/datasets/m2a_validation_dl/Mask2Anomaly/best_contrastive.pth \
  --images 'external/DetSeg-official/data/FS_LostFound/images/*.png' --out-dir results/score_maps/m2a/FS_LostFound
# (RoadAnomaly frames/*.webp, FS_Static images/*.jpg, SMIYC_Anomaly_val images/*.png, SMIYC_Obstacle_val images/*.webp)

# E2 / E3 (detseg env, external/DetSeg-official)
python tools/test.py ../../configs/detseg-r_m2a_$ds.py $CK --work-dir work_dirs/E2/$ds --cfg-options model.box_score_thr=1.1
python tools/test.py ../../configs/detseg-r_m2a_$ds.py $CK --work-dir work_dirs/E3/$ds
```

## 4. 결과

### 4.1 E2: Mask2Anomaly 단독 — 논문 재현 여부

| Dataset | AUROC 논문 / 우리 | AP 논문 / 우리 | FPR95 논문 / 우리 | IoU(탐색) 논문 / 우리 | 판정 |
|---|---|---|---|---|---|
| Road Anomaly | 96.57 / 95.87 | 79.70 / 82.05 | 13.45 / 15.73 | **58.43 / 58.41** | 재현됨 |
| FS L&F | 95.41 / 95.40 | 69.46 / 69.44 | 9.31 / 9.31 | **34.35 / 34.33** | 재현됨 (거의 동일) |
| FS Static | 98.18 / 98.35 | 82.37 / 90.01 | 7.20 / 1.91 | 60.51 / 45.56 | 버전 불일치 (참고) |
| SMIYC Anomaly val | – | – | – | **83.74 / 84.37** | 재현됨 |
| SMIYC Obstacle val | – | – | – | **67.13 / 67.17** | 재현됨 |

**→ 기존 방법은 4개 데이터셋에서 논문 수치를 거의 그대로 재현했다.** 우리 데이터(M2A 배포본)와 평가 지표가 논문과 같다는 강한 근거다.

### 4.2 E3: Mask2Anomaly + DetSeg-R

| Dataset | AUROC 논문 / 우리 | AP 논문 / 우리 | FPR95 논문 / 우리 | IoU(탐색) 논문 / 우리 |
|---|---|---|---|---|
| Road Anomaly | 98.33 / 98.45 | 87.57 / 88.27 | 5.7 / 8.92 | 65.27 / 70.18 |
| FS L&F | 96.79 / 95.97 | 78.15 / **70.84** | 4.84 / **9.39** | 35.46 / **31.15** |
| FS Static | 99.42 / 99.60 | 93.12 / 91.44 | 1.71 / 3.84 | 75.61 / 47.25 |
| SMIYC Anomaly val | – | – | – | 82.51 / **71.15** |
| SMIYC Obstacle val | – | – | – | 72.83 / 72.42 |

### 4.3 DetSeg-R의 개선폭 (E3 − E2) — 논문 vs 우리

| Dataset | AP 변화 논문 / 우리 | FPR95 변화 논문 / 우리 | IoU 변화 논문 / 우리 | 판정 |
|---|---|---|---|---|
| Road Anomaly | +7.87 / **+6.22** | −7.75 / **−6.81** | +6.84 / **+11.77** | ✅ 개선 재현 |
| FS L&F | +8.69 / **+1.40** | −4.47 / **+0.08** | +1.11 / **−3.18** | ❌ 개선 거의 없음, IoU는 악화 |
| FS Static | +10.75 / +1.43 | −5.49 / +1.93 | +15.1 / +1.69 | (버전 불일치, 참고) |
| SMIYC Anomaly val | – | – | −1.23 / **−13.22** | ❌ 크게 악화 |
| SMIYC Obstacle val | – | – | +5.70 / **+5.25** | ✅ 개선 재현 |

## 5. 해석

1. **기준점이 맞다.** M2A 단독이 FS L&F에서 소수 둘째 자리까지 논문과 같다 (AP 69.44 vs 69.46, FPR95 9.31 vs 9.31, IoU 34.33 vs 34.35). 따라서 E1의 FS L&F 불일치(DetSeg-S 14.26 vs 32.36)는 **데이터나 평가 방식 문제가 아니라 DetSeg 쪽(공개 가중치·코드)의 문제**다
2. **DetSeg-R도 DetSeg-S와 같은 곳에서 실패한다.** RA와 SMIYC Obstacle에서는 논문만큼 개선되지만, FS L&F에서는 개선이 거의 사라지고(AP +1.40, 논문 +8.69) IoU는 오히려 떨어진다. DetSeg-S와 DetSeg-R은 같은 OoD box를 쓰므로, **원인은 box 생성 단계(검출 + IE-NMS)**라는 것이 두 실험에서 일관되게 보인다
3. **SMIYC Anomaly에서 DetSeg-R이 크게 나빠진다** (IoU 84.37 → 71.15, AP 95.49 → 81.98). 논문도 이 데이터셋에서는 소폭 악화(−1.23)를 보고했고, 원인을 "box 안의 물체 아닌 영역이 새 오탐이 된다"고 설명했다(4.2절). 우리는 그 정도가 훨씬 크다. 10장뿐이라 이미지 한두 장의 영향일 수 있어 A2에서 이미지별로 확인한다
4. **FS Static은 M2A 단독도 IoU가 논문과 15포인트 차이**(45.56 vs 60.51) → 데이터 버전 차이가 확실하다. FS Static은 이후 참고로만 쓴다
5. DetSeg-R의 FPR95는 RA에서 논문보다 덜 줄었다(8.92 vs 5.7). M2A 단독 FPR95도 논문보다 높게 시작했다(15.73 vs 13.45)

## 6. 결과 파일

- 로그: `results/E2/{dataset}.log`, `results/E3/{dataset}.log`, score map 생성 로그 `results/E2/{Dataset}.log`
- score map: `results/score_maps/m2a/` (git 제외. `src/m2a_save_scores.py`로 재생성)
- 참고: FS L&F 첫 실행은 두 지표의 결과 키 이름이 겹쳐 지표 출력 뒤에 종료 오류가 났다 (값은 로그에 정상 출력). 이후 config에 prefix를 추가해 해결
