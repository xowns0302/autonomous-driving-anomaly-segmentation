# 02. 실험 범위와 계획

- 근거: [01_paper_summary.md](01_paper_summary.md), 공식 repo `external/DetSeg-official` (HEAD `a1aca17b`)
- 상태: **확정 (2026-10-16).** 결정 내용은 7절

---

## 1. 프로젝트 목표

1. **재현:** 공개 코드·가중치로 논문의 핵심 주장(DetSeg-S, DetSeg-R)이 우리 환경에서 성립하는지 확인한다
2. **분석:** 논문 수치와 다른 부분과 실패 사례의 원인을 근거로 밝힌다
3. **개선:** 분석에서 나온 약점을 재학습 없이 개선하고, 튜닝에 쓰지 않은 데이터로 검증한다
4. **보고:** 위 과정을 보고서·발표로 정리한다

비교 원칙: **모든 비교는 같은 환경 안에서 한다.** 기준은 "우리 환경의 baseline"이고, 논문 수치는 참고값이다.

---

## 2. 논문 실험 항목별 범위 판단

| 논문 항목 | 필요한 것 | 우리 환경에서 가능? | 판단 |
|---|---|---|---|
| **Table 2: DetSeg-S IoU** (RA, FS L&F, FS Static, SMIYC val) | 공개 가중치, SAM, 데이터 | ✅ 데이터 이미 확보 | **필수** |
| **Table 6: DetSeg-S ablation** (UN query / IE-NMS on·off) | 추론 시 끄기만 하면 됨 | ⚠️ 부분 가능 ("objects" prompt baseline, IE-NMS 끄기는 가능. attention mask on·off는 가중치가 한 가지라 의미 있게 재현 불가) | **필수 (부분)** |
| **Table 1: 기존 방법 vs + DetSeg-R** (AUROC/AP/FPR95) | **기존 방법의 score map** | ⚠️ 기존 방법마다 별도 repo·환경 필요 | **포함: Mask2Anomaly 1개** |
| **Table 2: 기존 방법 / + DetSeg-R IoU** (oracle threshold) | 위 score map | 위와 동일. 평가 도구(`OracleThresholdAnomalyIoUMetric`) 있음 | 위 결정을 따름 |
| IE-NMS bar **보정 절차** | Cityscapes(정답 box) 일부 | ⚠️ Cityscapes 가입·승인 필요 | **포함** |
| Table 5: DetSeg-R ablation (내부 RbA seg branch) | seg branch를 Cityscapes로 학습 (5,000 iter) | ⚠️ Cityscapes + 학습 필요, 공개 가중치에 없음 | 제외 (선택) |
| Table 3: SMIYC test | 공식 서버 제출 | ⚠️ 외부 제출 | 제외 |
| Table 4 + VPHM: LostAndFound 비디오 | 대용량 데이터 + VPHM 직접 구현 | ⚠️ 구현 필요 | 제외 (선택) |
| COCO fine-tuning (학습 재현) | COCO 19GB, A100 2장 규모 | ❌ 비용 대비 의미 낮음 (공개 가중치가 그 결과물) | 제외 |

### 2.1 DetSeg-R의 제약

DetSeg-R은 **혼자 동작하지 않는다.** 기존 방법이 만든 픽셀 score map이 있어야 한다.

| 기존 방법 | score map 생성 스크립트 (DetSeg repo) | 가중치 | 필요한 환경 | 비용 |
|---|---|---|---|---|
| **Mask2Anomaly** | 없음 (M2A repo를 직접 세팅) | ✅ 이미 받음 (`~/datasets/m2a_validation_dl/Mask2Anomaly/best_contrastive.pth`) | detectron2 + Mask2Former | 중 |
| **RPL+CoroCL** | ✅ `tools/baselines/save_energy_scores.py` | 다운로드 스크립트 있음 (약 0.9GB) | RPL repo | 중 |
| PEBAL | ✅ 같은 스크립트 | 다운로드 스크립트 있음 | PEBAL repo | 중 |
| UNO | ✅ `tools/baselines/save_uno_scores.py` | 다운로드 스크립트 있음 (약 2.5GB×2) | detectron2 + Mask2Former | 중~상 |
| RbA | 없음 | 별도 | detectron2 | 중~상 |

- 각 방법은 **자기 repo와 별도 conda 환경**이 필요하다 (detectron2 CUDA 빌드 등). 세팅 실패 위험이 있다
- score map을 한 번 만들어 두면, DetSeg-R과 우리 개선은 **같은 score map 위에서 반복 실험**할 수 있다
- 우리 IE-NMS 관련 개선은 DetSeg-R에도 같은 detector를 쓰므로 **S·R 양쪽에 적용**된다

### 2.2 범위 선택지

| 선택지 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **A. DetSeg-S만** | Table 2(DetSeg-S), Table 6(부분) | 바로 시작, 위험 낮음 | 논문 절반(DetSeg-R) 미검증, 기존 방법과의 비교 없음 |
| **B. S + R (기존 방법 1개)** ← 추천 | A + **Mask2Anomaly** 단독 / + DetSeg-R (Table 1·2) | "기존 방법 → DetSeg → 우리 개선" 3단 비교 완성. M2A는 가중치 보유, 논문 대표 비교 대상 | M2A 환경 세팅 (반나절~1일) |
| C. S + R (기존 방법 2개) | B + **RPL+CoroCL** | 성격이 다른 두 방법에서 일반성 확인 | 세팅 1회 더 |

**추천: B로 시작하고, 시간이 되면 C로 확장한다.**
- M2A가 세팅에 실패하면 RPL로 대체한다 (DetSeg repo에 스크립트가 있어 더 쉬움)

---

## 3. 데이터셋 계획

| Dataset | 장수 | 보유 | 용도 | 비고 |
|---|---|---|---|---|
| Road Anomaly | 60 | ✅ (EPFL WebP) | 재현·평가 | GT anomaly=2 → `UnifyGT({0:0, 2:1})` 필수 |
| FS L&F | 100 | ✅ (M2A 배포본) | 재현·평가 | |
| FS Static | 30 | ✅ (M2A 배포본) | 재현·평가 (참고) | **10장에 OoD 없음** → 이미지 평균 IoU 상한 66.7. 논문 버전과 다를 가능성 큼. 결과는 참고로만 |
| SMIYC Obstacle val | 30 | ✅ | 재현·평가 | 논문 Table 2 열 있음 |
| SMIYC Anomaly val | 10 | ✅ | 재현·평가 | 논문 Table 2 열 있음 |
| Cityscapes (일부) | – | ❌ (사용자 가입 대기) | IE-NMS bar 보정 | gtFine val(정답 box 생성용) + leftImg8bit val |

**튜닝/평가 분리 원칙 (개선 단계):**
- 개선의 하이퍼파라미터는 **보정용 데이터(Cityscapes) 또는 지정한 튜닝셋에서만** 고른다
- 최종 수치는 튜닝에 쓰지 않은 데이터에서 **한 번만** 측정한다
- 튜닝 = FS L&F + RA, 검증 = SMIYC Obstacle + Anomaly val (7절)

---

## 4. 평가 프로토콜

| 대상 | 지표 | 구현 |
|---|---|---|
| DetSeg-S | IoU (논문 정의: 이미지별 IoU 평균), F1 | `BinaryMaskAnomalyIoUMetric` |
| 기존 방법, + DetSeg-R | AUROC, AP, FPR95 | `AnomalyMetricLoad` |
| 기존 방법, + DetSeg-R | 최적 threshold IoU (0.01 간격 탐색) | `OracleThresholdAnomalyIoUMetric` |
| 보조 (분석용) | 전체 픽셀 IoU, box recall(면적별), sIoU/PPV/mean F1 근사 | 우리 스크립트 `src/eval_detseg_s_diagnostics.py` |

**재현 판정 기준 (실행 전 고정):** 논문과 차이 ≤ 3 → 재현됨 / 3~10 → 부분 재현 / > 10 → 불일치
(AUROC/AP/FPR95는 같은 기준을 %p 단위로 적용)

---

## 5. 실험 단계

### Phase 0. 준비
| ID | 내용 | 산출물 |
|---|---|---|
| P0-1 | 환경 점검 (conda `detseg`, GPU, 버전 기록) | `docs/03_environment_and_data.md` |
| P0-2 | 데이터 정리 (SMIYC val 연결, 라벨 값·장수 확인) | 같은 문서 |
| P0-3 | Mask2Anomaly 환경 세팅 + score map 생성 (5개 평가셋) | 같은 문서, score map 폴더 |
| P0-4 | Cityscapes val 확보 (사용자 가입 후) | 같은 문서 |

### Phase 1. 논문 재현 (공개본 그대로, 수정 없음)
| ID | 실험 | 논문 대응 |
|---|---|---|
| E1 | DetSeg-S IoU — RA, FS L&F, FS Static, SMIYC Anomaly·Obstacle val | Table 2 마지막 행 |
| E2 | 기존 방법 단독: AUROC/AP/FPR95 + 최적 threshold IoU | Table 1·2 기존 방법 행 |
| E3 | 기존 방법 + DetSeg-R: 같은 지표 | Table 1·2 "& DetSeg-R" 행 |
| E4 | DetSeg-S ablation: "objects" prompt만 / UN query + IE-NMS 끔 / 전체 | Table 6 (부분) |

### Phase 2. 분석
| ID | 내용 |
|---|---|
| A1 | 논문 대비 차이 정리, 판정 |
| A2 | 실패 진단: 오탐 출처(ID/UN branch), IE-NMS 작동 여부, 작은 물체 손실 단계 |
| A3 | IE-NMS bar를 **논문 방식으로 보정** (Cityscapes) → 공개 하드코딩 값과 비교 |
| A4 | DetSeg-R에서 같은 현상이 있는지 (box 품질이 score map 개선에 주는 영향) |

### Phase 3. 개선
| ID | 내용 |
|---|---|
| I1 | A2·A3 결과에서 개선 후보 선정 (아래 사전 탐색 결과 참고) |
| I2 | 튜닝셋에서 설정 결정 → 고정 |
| I3 | 검증셋에서 DetSeg-S·DetSeg-R 모두 원본 vs 개선 1회 비교 |
| I4 | ablation (구성 요소별 기여) |

### Phase 4. 보고서
| ID | 내용 |
|---|---|
| R1 | 결과 표·그림 정리 |
| R2 | 보고서·발표 자료 작성 |

---

## 6. 사전 탐색에서 이미 확인된 사실 (2026-10-12~13, git 브랜치 `exp/detseg-s-reproduction`)

이전 작업은 계획 없이 진행했으므로 수치는 **Phase 1에서 다시 측정**한다. 다만 아래 사실은 코드·데이터를 직접 확인한 것이라 계획에 반영한다.

| 사실 | 근거 | 계획에 미치는 영향 |
|---|---|---|
| 공개 checkpoint에 seg branch 가중치 없음 | checkpoint key 확인 | DetSeg-R internal(Table 5) 불가 → external만 |
| 공개 코드는 논문 약 9개월 뒤 재작성, 가중치 재학습 | git 이력 (`7953dc8e`, README) | 논문 수치와 차이가 날 수 있음. 같은 환경 비교 원칙의 이유 |
| IE-NMS bar가 하드코딩 (0.2/0.5), 보정 절차 없음 | `models.py:522` | A3 (논문 방식 보정) |
| query-text attention mask 미적용 | `models.py:1129`, decoder 코드 | 분석 항목. 연구 당시 코드도 동일 |
| DetSeg-S FS L&F 약 14, RA 약 76 (논문 32.36 / 68.35) | 이전 실행 | E1에서 재측정 |
| FS L&F 오탐 대부분이 차량 위, ID box 점수가 bar 0.5 미달로 IE-NMS 미작동 | 이전 진단 | A2에서 재확인 |
| 작은 물체는 objectness 점수 미달·후보 없음으로 손실 (IE-NMS 영향 작음) | 이전 진단 | A2에서 재확인, 개선 후보 |
| 탐색한 개선 후보: bar 재설정, 크기별 threshold, detector 자체 road box로 재점수화 | 이전 탐색 (튜닝/검증 분리 없음) | I1 후보. **크기별 threshold는 τ 고정 철학과 충돌하므로 후순위** |

---

## 7. 결정 사항 (2026-10-16 확정)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 범위 | **B: DetSeg-S + DetSeg-R, 기존 방법 1개 (Mask2Anomaly).** M2A 세팅 실패 시 RPL+CoroCL로 대체. 시간 남으면 C로 확장 |
| 2 | IE-NMS bar 보정 | **논문 방식으로 복원한다.** Cityscapes gtFine val 확보 (사용자 가입: https://www.cityscapes-dataset.com) |
| 3 | 튜닝/검증 분리 | **튜닝 = FS L&F + Road Anomaly, 검증 = SMIYC Obstacle val + Anomaly val.** SMIYC val은 재현(Phase 1)에는 쓰지만, 개선 설정을 고정하기 전까지 개선 실험에는 쓰지 않는다 |
| 4 | VPHM·LostAndFound 비디오, Table 5 내부 seg branch, SMIYC test 서버, COCO 재학습 | **제외** |
| 5 | 최종 threshold τ | **0.2 고정 유지** (논문 핵심 주장). τ를 바꾸는 개선은 별도 분석으로만 다룬다 |
