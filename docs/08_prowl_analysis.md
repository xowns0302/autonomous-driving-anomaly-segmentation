# 08. 관련 논문 분석: Finding Dino (PROWL)과 DetSeg 결합 가능성

> Sinhamahapatra, Schwaiger, Bose, Wang, Roscher, Günnemann. **Finding Dino: A Plug-and-Play Framework for Zero-Shot Detection of Out-of-Distribution Objects Using Prototypes.** WACV.
> 논문에 코드 링크는 명시되어 있지 않다.

---

## 1. 핵심 요약

**"아는 물체의 대표 특징(prototype)을 몇 개만 모아 두고, 어디에도 안 닮은 영역을 이상 물체로 본다."** 학습이 전혀 없다 (zero-shot, 추론만).

| 단계 | 내용 |
|---|---|
| 특징 추출 | **DINOv2 ViT-S/14** (자기지도 학습, 1.42억 장, 가중치 고정)로 이미지를 14px 패치 단위 특징 지도로 |
| prototype bank (오프라인) | 아는 class(도로 장면은 Cityscapes 19개)마다 정답 mask가 있는 물체 **L = 5~20개**의 특징 평균 → class별 prototype 목록 (식 1) |
| prototype 매칭 | 테스트 이미지의 픽셀(패치)마다 각 class prototype과의 **코사인 유사도 최대값** → class별 heatmap (식 2~4) |
| OOD 판정 | INCS = 1 − (이미지 안에서 min-max 정규화한 최대 유사도). INCS > t(기본 0.55)이면 이상 픽셀 (식 5) |
| 정제 (refinement) | 비지도 물체 mask(**CutLER** 검출기 또는 STEGO)를 구하고, **mask 안 픽셀의 과반이 이상이면 그 mask 전체를 이상 물체로** |

### 결과 (논문 Table 1·2, SMIYC는 validation)

| | RoadAnomaly21 | FS Static | RoadObstacle21 |
|---|---|---|---|
| PROWL + CutLER IoU (고정 threshold) | 75.22 | 64.79 | 49.16 |
| PROWL + CutLER AUPR / FPR | 75.25 / 1.75 | 70.27 / 8.21 | 73.53 / 5.58 |
| (참고) DetSeg-S IoU, DetSeg 논문 Table 2 | 75.82 | 83.34 | 81.58 |
| (참고) DetSeg-S + C5 IoU, 우리 I4 | 78.06 | – | 80.93 |

- PROWL 단독은 DetSeg-S보다 약하다. 특히 작은 장애물(RoadObstacle 49.16)에 약하다
- 대신 **학습이 없고, 이상 물체 데이터(auxiliary OOD)도 쓰지 않으며, 다른 도메인(철도·해상·인도 도로)으로 쉽게 옮겨진다**

### 논문이 밝힌 약점
- 픽셀 단위 판정은 배경(식생·건물)에도 오탐이 많다 → 물체 mask(CutLER)로 정제해야 한다
- CutLER threshold의 최적값이 데이터셋마다 다르다 (RA 0.7, FS Static 0.5, RO 0.2, Fig. 5b). 작은 물체는 낮은 threshold가 필요하다
- 작은 물체·먼 물체에서 성능이 떨어진다

---

## 2. DetSeg와 비교: 서로의 약점을 메운다

| | DetSeg | PROWL |
|---|---|---|
| **물체 찾기** | **강함**: UN query + SAM으로 물체 box·mask를 잘 만든다 | 약함: CutLER에 의존, threshold가 데이터셋마다 다름 |
| **"아는 물체인가" 판정** | **약함**: Grounding DINO의 class 점수 하나에 의존 → 이상 물체를 트럭으로 착각 (A2, I1) | **강함**: 19개 class와 직접 닮은 정도를 잼. 어느 class와 얼마나 닮았는지 알려 줌 |
| 학습 | UN query를 COCO로 fine-tuning | 없음 (prototype은 정답 mask 몇 개) |
| 속도 | 검출 ~200 ms + SAM | DINOv2 ViT-S는 가벼움 |

→ **"DetSeg가 물체를 찾고, PROWL이 그 물체가 아는 물체인지 판정한다."** PROWL 논문이 CutLER로 하던 "물체 mask 만들기"를 DetSeg box + SAM이 대신하는 구조다.

---

## 3. 우리 실험에 반영하는 방법

### 방법 A: C5의 확인 신호를 M2A → PROWL prototype으로 교체 (C6)

C5는 "ID box가 OoD box를 지우려 할 때, 다른 모델이 '모르는 물체'라고 하면 막는다"였다. 그 다른 모델을 M2A 대신 PROWL로 바꾼다.

| | C5 (M2A) | C6 (PROWL prototype) |
|---|---|---|
| 확인 신호 | box 안 M2A 이상 점수 평균 | box(또는 SAM mask) 안 **19개 prototype과의 최대 유사도** (낮을수록 모르는 물체) |
| 이상 데이터 사용 | M2A는 COCO 물체를 붙여 넣은 이미지로 추가 학습 (auxiliary OOD 데이터) | **없음** |
| 비용 (T2 측정) | M2A 약 400~500 ms (2회 추론) | DINOv2 ViT-S 1회 (훨씬 가벼울 것으로 예상, 측정 필요) |
| 추가 장점 | – | **class를 지목해 확인 가능**: ID box가 "truck"이라는데 그 영역이 truck prototype과 안 닮았으면 모순 → 보호 |

### 방법 B: 남은 OoD box 중 "아는 물체와 너무 닮은 것"을 버리기 (C7)

- I4 사전 분석에서 M2A로 이 방향(남은 오탐 제거)은 RA에서 AUC 0.434로 실패했다. M2A가 Cityscapes(독일 도로) 밖의 인터넷 사진에 약하기 때문이다
- DINOv2는 다양한 1.42억 장으로 배워 환경 변화에 강하다고 보고된다 (PROWL은 인도 도로 IDD에도 적용). **이 방향이 될 가능성이 M2A보다 높다**
- 겨냥: "정상 물체를 이상 물체로 판단"하는 오탐 (사용자가 지적한 두 번째 문제)

### 방법 C: SAM mask 단위 과반 투표 (PROWL 정제 단계 그대로)

- DetSeg-S의 SAM mask마다 픽셀의 과반이 "아는 물체와 닮음"이면 제거. B를 box 대신 mask로 하는 것. B와 함께 비교

---

## 4. 우리 환경에서 필요한 것

| 항목 | 상태 |
|---|---|
| DINOv2 ViT-S/14 가중치 | 다운로드 필요 (공개, 약 90MB). `detseg` 환경의 transformers 4.46에 `Dinov2Model` 있음 |
| prototype용 정답 mask | Cityscapes **train** split. gtFine zip에 이미 있음 (val만 풀어 둠), 이미지 zip에도 있음 → class당 5~20개 물체만 꺼내면 됨. 평가셋과 겹치지 않음 |
| stuff class(road 등) prototype | 개별 물체 라벨이 없으므로 semantic mask 영역을 사용 (PROWL도 픽셀 영역 기준) |
| 구현 | 논문에 코드가 없어 직접 구현 (식 1~5, 수십 줄 수준) |
| 재학습 | 없음 |

## 5. 위험

1. **작은 물체:** DINOv2는 14px 패치 단위다. FS L&F의 작은 장애물은 패치 1~2개 크기라 판정이 거칠 수 있다 → 입력을 키워 패치를 촘촘히 할지 확인 필요 (DINOv2는 C4의 Grounding DINO와 달리 해상도 변화에 비교적 강하다고 알려져 있으나 확인 필요)
2. **이미지별 min-max 정규화 (식 5):** 이상 물체가 없는 이미지도 가장 낮은 유사도 영역이 "이상"이 된다. box 단위 확인에는 정규화 전 원래 유사도를 쓰는 편이 안전하다 (우리 설계 선택)
3. 데이터 누수 방지: prototype은 Cityscapes train에서만, 설정값은 튜닝셋(FS L&F + RA)에서만
4. 효과는 실험 전에는 모른다 → **먼저 사전 분석** (I4와 같은 방식: prototype 신호가 진짜 이상 물체와 오탐을 얼마나 구분하는지 AUC)

## 6. 진행 제안

1. DINOv2 ViT-S/14 받기, Cityscapes train에서 class당 20개 물체로 prototype bank 만들기
2. **사전 분석 (튜닝셋만):** OoD 후보 box / SAM mask마다 prototype 최대 유사도를 계산 → (a) IE-NMS가 지운 후보, (b) 남은 후보에서 진짜 vs 오탐 구분 AUC. M2A(0.946 / 0.716, 0.690 / 0.434)와 비교
3. 사전 분석이 좋으면 C6(보호)·C7(제거)을 계획 문서에 등록 → 튜닝 → 검증셋 1회 → 시간 측정 (T2 규칙)
