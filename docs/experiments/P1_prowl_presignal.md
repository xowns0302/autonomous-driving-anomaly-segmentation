# P1. PROWL prototype 신호 사전 분석

- 날짜: 2026-10-18
- 단계: Phase 3 (개선) 사전 분석. 배경: [08_prowl_analysis.md](../08_prowl_analysis.md)
- 결론: **PROWL prototype 신호가 M2A보다 모든 비교에서 진짜 이상 물체와 오탐을 더 잘 구분하고, 약 5~9배 빠르다.** 다만 Road Anomaly에서 "남은 오탐 제거" 방향은 여전히 구분하지 못한다

## 1. 준비

| 항목 | 내용 |
|---|---|
| 특징 추출기 | DINOv2 ViT-S/14 (`facebook/dinov2-small`, HuggingFace, 22M 파라미터, 85MB). 받는 법: `AutoModel.from_pretrained('facebook/dinov2-small')` → `~/.cache/huggingface/`에 자동 저장 |
| 입력 크기 | 짧은 변 518, 양 변 14의 배수 (FS L&F 2048×1024 → 1036×518 = 74×37 패치) |
| prototype bank | Cityscapes **train**에서 class당 20개 (총 380개, 133장). thing class는 개별 물체 mask, stuff class는 한 이미지의 class 영역. 면적 4,000px 이상, seed 0, 이미지당 class별 최대 1개 (`src/prowl_bank.py`) |
| prototype | mask가 절반 이상 덮는 패치의 특징 평균 → L2 정규화 (논문 식 1) |
| 확인 | class 간 가장 닮은 쌍이 상식과 맞음: wall↔fence 0.90, bus↔train 0.87, person↔rider 0.85, truck↔bus 0.83 |

## 2. 방법 (`src/analyze_prowl_box_signal.py`)

I4의 M2A 사전 분석(`src/analyze_m2a_box_signal.py`)과 **같은 후보, 같은 분류**:
- 논문 방식 bar, OoD 후보(label 0, 점수 > 0.2), 튜닝셋(FS L&F + RA)만
- 진짜 = 정답 물체와 box IoU ≥ 0.1, 오탐 = 그 외
- IE-NMS가 **지운** 후보 / **남긴** 후보로 나눔

이상 신호 (높을수록 이상):
- **proto_box** = 1 − max_k,l cos(box 안 패치 특징 평균, prototype)
- **proto_patch** = 1 − (box 안 패치별 "가장 닮은 prototype과의 유사도"의 평균)
- **m2a** = box 안 Mask2Anomaly 점수 평균 (I4와 같은 값, 비교용)

## 3. 결과: 진짜 vs 오탐 구분력 (AUC, 1.0 = 완벽, 0.5 = 동전 던지기)

| Dataset | 후보 그룹 | 진짜 / 오탐 | **proto_box** | **proto_patch** | m2a (I4) |
|---|---|---|---|---|---|
| FS L&F | IE-NMS가 지운 후보 | 107 / 2,084 | **0.985** | 0.982 | 0.945 |
| FS L&F | IE-NMS가 남긴 후보 | 89 / 234 | **0.941** | 0.926 | 0.681 |
| RA | IE-NMS가 지운 후보 | 277 / 460 | 0.794 | **0.811** | 0.715 |
| RA | IE-NMS가 남긴 후보 | 203 / 224 | 0.481 | 0.473 | 0.432 |

신호 분포 (proto_box, 10 / 50 / 90 백분위):

| Dataset | 그룹 | 진짜 | 오탐 |
|---|---|---|---|
| FS L&F | 지운 후보 | 0.38 / 0.50 / 0.64 | 0.16 / 0.25 / 0.35 |
| FS L&F | 남긴 후보 | 0.39 / 0.51 / 0.64 | 0.23 / 0.32 / 0.44 |
| RA | 지운 후보 | 0.37 / 0.54 / 0.72 | 0.27 / 0.38 / 0.54 |
| RA | 남긴 후보 | 0.39 / 0.57 / 0.72 | **0.33 / 0.62 / 0.75** |

시간: DINOv2 + prototype 비교 **53 ms (RA) ~ 94 ms (FS L&F) / 이미지** (M2A 약 400~500 ms, T2)

## 4. 해석

1. **"지워진 후보 보호"(C6) 방향: M2A보다 확실히 좋다.** FS L&F 0.945 → 0.985, RA 0.715 → 0.81. C5와 같은 규칙에 신호만 바꾸면 더 정확하고, 이상 물체 데이터로 학습한 모델(M2A)도 필요 없다
2. **"남은 오탐 제거"(C7) 방향: FS L&F에서는 처음으로 가능해졌다** (M2A 0.681 → 0.941). FS L&F의 남은 오탐은 대부분 Cityscapes에 있는 물체(울타리, 기둥, 차 일부)라 prototype과 닮았다
3. **RA의 남은 후보는 여전히 구분 불가** (0.48). RA 오탐의 신호(중앙 0.62)가 오히려 진짜(0.57)보다 높다. RA는 인터넷 사진(농장, 시골 길 등)이라 오탐 대상 자체가 **Cityscapes에 없는 모습**이다. "아는 물체 = Cityscapes 19개"라는 정의의 한계로, 어떤 Cityscapes 기반 신호(M2A 포함)로도 풀기 어렵다
4. **속도:** M2A 대비 약 5~9배 빠르다. C5를 C6로 바꾸면 C5의 가장 큰 비용(M2A)이 사라진다. 패치 비교(19 class × 20개)를 줄이면 더 빨라질 여지가 있다
5. box 평균(proto_box)과 패치 평균(proto_patch)은 비슷하다. 계산이 단순한 proto_box를 기본으로 한다

## 5. 다음: 계획 등록 제안

| ID | 규칙 | 신호 | 후보값 (튜닝셋에서 선택) | 근거 |
|---|---|---|---|---|
| C6 | ID box는 proto_box ≥ t6인 OoD box를 지우지 못한다 (C5의 신호 교체) | proto_box | t6 ∈ {0.35, 0.40, 0.45, 0.50} | 지운 후보에서 진짜 10백분위 0.37~0.38, 오탐 90백분위 0.35~0.54 |
| C7 | 남은 OoD box 중 proto_box < t7이면 버린다 | proto_box | t7 ∈ {0.25, 0.30, 0.35} | FS L&F 남긴 후보에서 오탐 중앙 0.32, 진짜 10백분위 0.39. RA에서 위험 |

## 6. 결과 파일

- `results/P1/bank_objects.json` (선택한 물체 380개), `bank_objects_images.txt`, `prototypes.npz`
- `results/P1/box_signal_auc.json`, `box_signal_rows_{fs_lostfound,road_anomaly}.npy`
