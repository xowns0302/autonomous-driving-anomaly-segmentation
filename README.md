# Autonomous Driving Anomaly Segmentation

자율주행 도로 장면의 이상 물체(OoD) 검출 졸업프로젝트.
주 논문 **DetSeg** (*Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes*, ICCV 2025)를 같은 환경에서 재현·분석하고, 재학습 없이 개선한다.

## 문서 (읽는 순서)

| 문서 | 내용 |
|---|---|
| [docs/01_paper_summary.md](docs/01_paper_summary.md) | 논문 핵심 정리 |
| [docs/02_scope_and_plan.md](docs/02_scope_and_plan.md) | 실험 범위, 단계, 평가 프로토콜 |
| [docs/03_environment_and_data.md](docs/03_environment_and_data.md) | 환경·데이터 |
| [docs/04_code_walkthrough.md](docs/04_code_walkthrough.md) | 코드 흐름 설명 (이미지 한 장의 처리 과정, 스크립트 역할) |
| [docs/05_improvement_plan.md](docs/05_improvement_plan.md) | 개선 계획 (Phase 3, 실행 전 확정) |
| [docs/06_decision_logic.md](docs/06_decision_logic.md) | 판단 기준 설명 (이미지 한 장의 숫자로) |
| [docs/07_full_pipeline_explained.md](docs/07_full_pipeline_explained.md) | 사진 한 장이 점수가 되기까지: 등장인물(모델)과 흐름 |
| [docs/experiments/](docs/experiments/) | 실험별 기록 (실행할 때마다 추가) |

## 구조

```text
docs/           계획·논문 정리·실험 기록 (md)
configs/        DetSeg 공식 config를 상속한 우리 config
src/            평가·진단 스크립트
results/        실험별 요약 결과 (summary.json 등, 실험 시작 후 생성)
assets/         그림
external/       공식 코드 (git 제외): DetSeg-official
```

데이터는 `~/datasets/`에 두고 `external/DetSeg-official/data/`에 symlink로 연결한다.
