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
| I1 | [억제 규칙 C1·C2](I1_suppression_rules.md) | **기각**: 두 규칙 모두 튜닝셋에서 크게 악화 (배경 box는 거대 오탐을 지우는 역할, objectness는 아는 물체에 더 높음) |
| I2 | [도로 문맥 재점수화 C3](I2_road_context.md) | **채택 안 함**: 튜닝 +0.54, 검증 개선 없음 (bar 보정이 도로 밖 오탐을 이미 제거) |
| I3 | [고해상도 추론 C4](I3_high_resolution.md) | **기각**: 1.25배 FS 24.26 → 10.72, 1.5배 → 0.32 (학습 해상도 밖에서 box 예측 붕괴) |
| I4 | [M2A로 확인하는 IE-NMS C5](I4_m2a_verified_ie_nms.md) | **성공**: 검증 SMIYC Obstacle 77.93 → 80.93, Anomaly 54.57 → 78.06 (DetSeg-S). FS L&F 32.67로 논문 수치 도달 |
| T1 | [처리 시간 분석](T1_latency_profile.md) | DetSeg-S 808 ms 중 SAM 54% (대부분 CPU 전·후처리), 검출 23%. DetSeg-R 227 ms (논문 225 ms) |
| T2 | [처리 시간 baseline](T2_latency_baseline.md) | 고정 측정 규칙. DetSeg-S 603~808 ms, DetSeg-S + C5 약 1.1~1.2 s, M2A + DetSeg-R + C5 667~857 ms |
| P1 | [PROWL prototype 신호 사전 분석](P1_prowl_presignal.md) | 지운 후보 구분 AUC FS 0.985 / RA 0.81 (M2A 0.945 / 0.715), 남은 후보 FS 0.941 (M2A 0.681), RA 0.48. M2A보다 5~9배 빠름 |
| I5 | [PROWL 신호로 보호(C6)·제거(C7)](I5_prowl_rules.md) | C6 **기각** (RA 하락). C7 선택: 튜닝 +2.35, 검증 +0.1 수준 (FS L&F 오탐 제거에만 효과) |
| I6 | [C5 + C7 조합 (C8)](I6_c5_c7_combined.md) | 성공 기준 충족. FS L&F 38.44 (최고), SMIYC Obstacle 81.03, SMIYC Anomaly 69.58 (C5 78.06보다 낮음). 가장 느림 (DetSeg-S 1.15~1.37 s) |
