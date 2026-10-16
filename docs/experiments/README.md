# 실험 기록

실험마다 파일 하나: `E1_detseg_s_reproduction.md` 같은 형식 (ID는 [02_scope_and_plan.md](../02_scope_and_plan.md) 5절 기준).

각 기록에 들어갈 항목:

1. 목적 (어떤 질문에 답하는가, 논문 어느 표에 대응하는가)
2. 설정 (config, checkpoint, 데이터, 바꾼 것)
3. 실행 명령
4. 결과 (표, 논문 수치와 비교, 판정)
5. 해석
6. 결과 파일 위치

| ID | 실험 | 상태 |
|---|---|---|
| E1 | [DetSeg-S 재현 (Table 2)](E1_detseg_s_reproduction.md) | 완료: RA·SMIYC 부분 재현, FS L&F·FS Static 불일치 |
| E2·E3 | [M2A 단독 vs + DetSeg-R (Table 1·2)](E2_E3_m2a_and_detseg_r.md) | 완료: M2A 단독은 논문과 거의 동일, DetSeg-R은 RA·Obstacle 개선 재현 / FS L&F·SMIYC Anomaly 실패 |
| E4 | [DetSeg-S ablation (Table 6 부분)](E4_detseg_s_ablation.md) | 완료: 논문과 같은 경향, IE-NMS 효과 큼. SMIYC Anomaly는 반대 |
| A2 | [실패 진단 (IE-NMS가 지우는 것·남기는 것)](A2_failure_diagnosis.md) | 완료: 과소 억제(차량 오탐) + 과잉 억제(SMIYC Anomaly 이상 물체 삭제) |
| A3 | [IE-NMS bar 논문 방식 보정 (Cityscapes)](A3_ie_nms_bar_calibration.md) | 완료: bar ≈ 0.1 (공개 0.5). FS L&F 14.26 → 24.26, SMIYC Anomaly 69.12 → 54.57 |
