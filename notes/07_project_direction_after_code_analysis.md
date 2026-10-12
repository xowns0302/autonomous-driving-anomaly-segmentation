# DetSeg 코드 분석 이후 프로젝트 방향

- 작성일: 2026-10-12
- 역할: [06 코드 분석](06_detseg_code_analysis.md)과 [08 baseline 재현 계획](08_detseg_s_baseline_reproduction_plan.md)을 잇는 **프로젝트 방향 확정 문서**
- 상태: 방향 확정. baseline 결과가 나오면 7절의 분기에 따라 갱신한다.

---

## 1. 프로젝트 방향 재정의

지금까지 한 단일 이미지 inference(`src/run_detseg_single_image.py`, road_debris threshold 비교)는 **환경 sanity check**였다. DetSeg가 로컬 WSL2 + RTX 4070 Ti에서 돌아가고 box가 나온다는 것을 확인한 것이 전부다.

이 프로젝트의 실제 핵심은 다음과 같다.

> **DetSeg-S baseline이 논문 Table 2를 재현하는지 확인하고, 재현 여부에 따라 box-level OoD 후보 생성 로직의 실패 지점을 진단한 뒤, small-object / IE-NMS / road-aware 후처리 개선 가능성을 탐색한다.**

즉 흐름은 "재현 → 진단 → 개선" 순서이고, 각 단계의 결과가 다음 단계를 결정한다.

---

## 2. 현재까지의 결론

| 대상 | 결정 | 근거 |
|---|---|---|
| 단일 이미지 heatmap | **실험 근거로 사용하지 않는다** | internal config의 heatmap은 학습되지 않은 seg branch에서 나온 값이다. box 결과만 유효하다 (06 문서 발견 #1) |
| DetSeg-R internal | **제외** | 배포 checkpoint에 `seg_decoder`, `neck_seg` 가중치가 없다 (checkpoint key 직접 확인). 쓰려면 Cityscapes로 seg branch를 따로 학습해야 한다 |
| DetSeg-R external | **후순위** | M2A, RbA 등 외부 방법의 anomaly score map(`other_score_results/...`)이 필요한데, repo에 없고 직접 생성해야 한다 |
| VPHM | **범위 제외** | 논문 3.3절에는 설명이 있지만, 배포 코드에는 실행 불가능한 `VanishingPointGuidedCost`만 있고 호출처도 없다. 비디오 데이터와 frame loop도 새로 구현해야 한다 (06 문서 발견 #2) |
| DetSeg-S | **baseline 재현 대상** | 외부 score map 없이 box → SAM mask로 바로 평가할 수 있다. 논문 Table 2·6이 DetSeg-S IoU로 보고되어 있어 비교 기준이 명확하다 |

---

## 3. 왜 baseline 재현이 먼저인가

1. **기준은 논문 Table 2의 DetSeg-S IoU다.**
   Road Anomaly 68.35 / FS L&F 32.36 / FS Static 83.34 / SMIYC Anomaly 75.82 / SMIYC Obstacle 81.58
2. **코드가 논문 설명과 다를 가능성이 두 군데 있다.**
   - **Attention mask**: 논문은 UN query가 ID class 이름을 보지 못하게 가린다고 설명한다. Table 6에서는 mask가 빠지면 FS L&F IoU가 32.36 → 7.29(IE-NMS도 없을 때)로 떨어진다. 그런데 코드에서는 `query_text_mask`가 decoder의 text cross-attention까지 전달되지 않는 것으로 보인다 (06 문서 발견 #3).
   - **IE-NMS bar**: 논문은 Cityscapes로 class별 bar를 보정한다고 쓴다(예: car 0.36, truck 0.58, bicycle 0.28). 코드는 `[0.2]*11 + [0.5]*8`로 단순화되어 있다.
3. **그래서 개선보다 재현 확인이 먼저다.** baseline이 논문과 다르면, 그 위에서 측정한 개선 효과는 해석할 수 없다.
4. **재현이 안 되는 것도 결과다.** 그렇다면 "논문 구현과 배포 코드의 차이"가 이 프로젝트의 중요한 분석 결과가 된다. 어느 경우든 보고서에 쓸 내용이 생긴다.

---

## 4. Baseline 재현 우선순위

| 순위 | Dataset | 선택 이유 |
|---|---|---|
| 1 | **FS L&F** (100장) | 작은 도로 장애물 위주라 개선 주제(작은 물체 보존)와 바로 연결된다. Table 2에서 DetSeg-S IoU가 32.36으로 가장 낮아 개선 여지가 크다. default config가 이미 이 dataset을 가리켜서 설정 변경 없이 바로 돌릴 수 있다 |
| 2 | **Road Anomaly** (60장) | 동물, 돌, 라바콘 등 물체 종류가 다양하고 도로 밖 물체도 있다. 나중에 road-aware 개선이 도로 밖 OoD를 해치는지 검증하는 데 필요하다 |
| 3 | **FS Static** | 논문 ablation(Table 5·6)의 3개 dataset 중 하나라 비교하려면 필요하다. 다만 논문이 데이터 버전 불일치를 언급해서 재현 난이도가 높다 |
| 4 | (선택) **SMIYC Obstacle val** | 작은 장애물과 다양한 날씨 조건. test 라벨은 비공개라 val만 직접 평가할 수 있다 |

실행 순서: FS L&F 하나로 파이프라인 전체를 먼저 통과시키고, Road Anomaly를 추가한 뒤 Table 2와 비교한다.

---

## 5. Road Anomaly / FS Static 주의사항

08 문서를 쓰면서 코드에서 새로 확인한 내용이다.

- ~~Road Anomaly는 `RoadAnomalyJPGDataset`을 써야 한다.~~ **(2026-10-13 정정)** EPFL은 WebP판(`RoadAnomaly.zip`)과 JPG판(`RoadAnomaly_jpg.zip`)을 둘 다 제공한다. 실제로는 WebP판을 받아 원래 클래스 `RoadAnomalyDataset`을 사용했다.
- **Road Anomaly는 `UnifyGT`가 필요하다 (확정: GT 값 `{0, 2}`).** 다른 공식 도구(box recall, DetSeg-R 평가)는 GT 값 `1 또는 2`를 OoD로 처리하지만, DetSeg-S의 metric(`BinaryMaskAnomalyIoUMetric`)은 `gt == 1`만 OoD로 본다. DetSeg-S config에는 `UnifyGT(label_map={0:0, 2:1})`가 주석 처리되어 있다. 라벨 값을 먼저 확인하고 필요하면 override config로 넣는다. 그대로 돌리면 IoU가 거의 0으로 나올 수 있다.
- **FS Static은 경로 설정 override가 필요하다.** DetSeg-S config의 prefix는 `images/labels_masks`로 남아 있지만, 다른 공식 도구들은 `original/labels`를 쓴다.
- **SAM 모델 크기가 논문에 명시되어 있지 않다.** 코드는 `facebook/sam-vit-base`를 쓴다. 논문이 더 큰 모델(ViT-H)을 썼다면 IoU 차이의 원인이 될 수 있다.

---

## 6. 판정 기준

결과를 보기 전에 고정한다 (08 문서 7절과 동일).

| 논문 IoU와의 차이 | 판정 |
|---|---|
| ≤ 3 | **재현됨** |
| 3 초과 ~ 10 이하 | **부분 재현** (배포 checkpoint 재학습에 따른 편차일 수 있음) |
| 10 초과 | **불일치** |

---

## 7. 재현 후 진행 방향

### 7.1 재현 성공 시 (재현됨 / 부분 재현)

1. **box 수준 지표 측정**: box recall, 이미지당 FP box 수, 면적 구간별 recall (`tools/eval_box_recall.py --min-area`)
2. **IE-NMS 전후 진단**: GT OoD와 매칭되는 UN box 중 IE-NMS에서 제거된 개수, 그리고 그 box를 제거한 ID class
3. **작은 물체 recall이 어디서 줄어드는지 분석**: encoder top-300 / IE-NMS / 최종 threshold 0.2 중 어느 단계인지
4. **진단 결과에 맞는 개선안을 골라 ablation**:
   - road-aware soft re-ranking
   - size-aware threshold
   - IE-NMS protection (높은 objectness 보호, 작은 물체 보호, stuff class 강화 제외)

> 이 프로젝트는 "IE-NMS가 진짜 OoD를 제거한다"를 전제로 하지 않는다. 2단계 진단에서 그런 사례가 드물면, 실제로 recall 손실이 생기는 단계로 개선 대상을 옮긴다.

평가 시 지켜야 할 것:
- recall만 보지 않고 FP를 함께 본다 (recall–FP 곡선)
- 하이퍼파라미터를 튜닝한 dataset과 최종 보고하는 dataset을 분리한다
- 결과는 dataset별로 따로 보고한다 (road-aware는 도로 밖 OoD가 있는 Road Anomaly에서 불리할 수 있음)

### 7.2 재현 실패 시 (불일치)

비용이 낮은 것부터 아래 순서로 점검한다 (세부 항목은 08 문서 8절).

1. dataset / GT 처리 (로드된 장수, 라벨 값, UnifyGT, void 처리)
2. metric 정의 (이미지 평균 IoU인지 전체 픽셀 IoU인지)
3. checkpoint mismatch (missing key, 재학습본 편차)
4. attention mask 미적용 가능성 (hook으로 확인 → mask 적용 패치와 비교)
5. IE-NMS bar 차이 (bar 스윕으로 민감도 확인)
6. SAM model size / transformers version (`sam-vit-huge`와 1회 비교)

4~6에서 원인이 확인되면, 그 분석 자체를 보고서의 "재현성 분석" 장으로 쓴다.

---

## 8. 최종 프로젝트 후보 제목

1. **DetSeg-S 재현성 분석 및 작은 도로 장애물 보존을 위한 Box-level 후처리 개선**
2. **자율주행 도로 장면 OoD 객체 탐지를 위한 DetSeg의 Box-level 후보 생성 로직 분석과 개선**
3. **Road-aware and Size-aware Post-processing for DetSeg-based Road OoD Object Detection**

최종 제목은 baseline 결과와 진단 결과를 본 뒤 고른다. 재현이 안 되면 1번처럼 "재현성 분석"이 들어간 제목이 맞고, 재현이 되고 개선 효과가 확인되면 2번이나 3번이 맞다.

---

## 9. 바로 다음 실행 계획

1. FS L&F 데이터 확보 (`images/`, `labels_masks/` 구조, 100장)
2. `external/DetSeg-official/data/FS_LostFound` symlink 생성
3. GT label 값 확인 (기대값 `{0, 1, 255}`, 08 문서 5.0절 스크립트)
4. DetSeg-S default config 실행 (08 문서 5.1절 명령)
5. Table 2의 FS L&F IoU **32.36**과 비교하고 6절 기준으로 판정
6. 결과를 `experiments/result_log.md`에 기록
