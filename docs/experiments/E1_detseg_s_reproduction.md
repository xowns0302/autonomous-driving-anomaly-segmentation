# E1. DetSeg-S 재현 (논문 Table 2 마지막 행)

- 날짜: 2026-10-16
- 단계: Phase 1 (재현). 공개 코드·가중치 그대로, 수정 없음

## 1. 목적

공개된 DetSeg-S가 우리 환경에서 논문 Table 2의 IoU를 재현하는가?

## 2. 설정

| 항목 | 값 |
|---|---|
| 모델 | `DetSegS` (Grounding DINO Swin-B + UN query + IE-NMS + SAM ViT-B) |
| 공식 config | `external/DetSeg-official/configs/detseg/detseg-s_swin-b.py` |
| 우리 config | `configs/detseg-s_{fs_lostfound,road_anomaly,fs_static,smiyc_anomaly,smiyc_obstacle}.py` (dataset 경로만 변경, Road Anomaly는 UnifyGT 2→1 추가) |
| checkpoint | `detseg_swin-b_coco_20260623-207453f5.pth` |
| 최종 threshold τ | 0.2 (코드 기본값 = 논문) |
| IE-NMS bar | 코드 기본값 (stuff 0.2 / thing 0.5) |
| 지표 | `BinaryMaskAnomalyIoUMetric`: 이미지별 IoU의 평균, F1(=Dice) 평균 |
| 환경·데이터 | [03_environment_and_data.md](../03_environment_and_data.md) |

## 3. 실행 명령

```bash
cd external/DetSeg-official && conda activate detseg && export PYTHONPATH=.
CK=ckpts/detseg_swin-b_coco_20260623-207453f5.pth
for ds in fs_lostfound road_anomaly fs_static smiyc_anomaly smiyc_obstacle; do
  python tools/test.py ../../configs/detseg-s_$ds.py $CK --work-dir work_dirs/E1/$ds
done
```

## 4. 결과

모든 실행 정상 종료, 전체 이미지 처리, checkpoint missing/unexpected key 경고 없음.

| Dataset | 장수 | 논문 IoU | **우리 IoU** | 차이 | 우리 F1 | 판정 |
|---|---|---|---|---|---|---|
| Road Anomaly | 60 | 68.35 | **76.46** | +8.11 | 84.07 | 부분 재현 (논문보다 높음) |
| FS L&F | 100 | 32.36 | **14.26** | −18.10 | 18.53 | **불일치** |
| FS Static | 30 | 83.34 | **33.51** | −49.83 | 41.41 | **불일치** (아래 주의) |
| SMIYC Anomaly val | 10 | 75.82 | **69.12** | −6.70 | 75.08 | 부분 재현 |
| SMIYC Obstacle val | 30 | 81.58 | **77.93** | −3.65 | 83.21 | 부분 재현 |

판정 기준 (02 문서 4절): |차이| ≤ 3 재현됨, 3~10 부분 재현, > 10 불일치

**OoD가 없는 이미지를 뺀 IoU** (OoD 없는 이미지는 지표 정의상 항상 0점이므로 정확히 계산된다):

| Dataset | OoD 없는 이미지 | OoD 있는 이미지만의 IoU |
|---|---|---|
| FS L&F | 1 / 100 | 14.40 |
| FS Static | 10 / 30 | **50.27** (= 33.51 × 30 / 20) |

## 5. 해석

1. **5개 중 3개는 부분 재현, 2개(FS L&F, FS Static)는 불일치.** 실패가 FS 계열 두 데이터셋에 몰려 있다
2. **FS Static은 데이터 버전 문제가 크다.** 우리 버전은 30장 중 10장에 OoD가 없어 이미지 평균 IoU의 상한이 66.7이다. 논문 83.34는 이 상한보다 높으므로 **논문은 다른 버전을 썼다**는 것이 확실하다. OoD 있는 이미지만 봐도 50.27로 여전히 낮다. FS Static은 판정에서 "버전 불일치로 비교 불가, 참고값"으로 둔다
3. **FS L&F는 데이터 문제가 아니다.** OoD 없는 이미지는 1장뿐이고 표준 100장 구성이다. 차이(−18.1)의 원인은 Phase 2(A2)에서 진단한다
4. **같은 설정을 두 번 실행해 같은 값이 나왔다** (FS L&F 14.26, RA 76.46, FS Static 33.51을 2026-10-12에도 동일하게 얻음) → 결과가 결정적(deterministic)이다
5. FS L&F와 SMIYC Obstacle은 둘 다 작은 장애물 데이터인데 결과가 크게 다르다 (14.26 vs 77.93). FS L&F에 특유한 요인(주변 차량이 많은 도시 장면 등)이 있는지 A2에서 본다

## 6. 결과 파일

- 로그: `results/E1/{dataset}.log`
- 원본 출력: `external/DetSeg-official/work_dirs/E1/{dataset}/` (git 제외)
