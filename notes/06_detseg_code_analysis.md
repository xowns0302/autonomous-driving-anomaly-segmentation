# DetSeg 핵심 로직 코드 분석

- 작성일: 2026-10-12
- 분석 대상: `external/DetSeg-official` (HEAD `a1aca17b`)
- 범위: 코드 정적 분석 + checkpoint key 확인. **코드 수정 없음.**
- 표기: 모든 경로는 `external/DetSeg-official/` 기준. 줄 번호는 위 커밋 기준.
- 「확인 필요」 = 코드만으로 확정할 수 없거나, 실행 로그/논문 원문으로 검증해야 하는 항목.

---

## 0. 먼저 알아야 할 핵심 발견 (TL;DR)

DetSeg의 실제 동작은 거의 전부 **`mmdet/detseg_utils/models.py` 한 파일**(1,769줄)에 있다. 나머지 `mmdet/`는 MMDetection 3.3 원본이다.

분석 중 발견한, 프로젝트 진행에 직접 영향을 주는 사실 4가지:

| # | 발견 | 근거 | 영향 |
|---|---|---|---|
| 1 | **지금까지 만든 anomaly heatmap은 학습되지 않은 seg branch에서 나온 값이다.** | 배포 checkpoint의 key에 `seg_decoder.*`, `neck_seg.*`가 없음 (직접 확인: backbone/neck/encoder/decoder/bbox_head/language_model/universal_* 만 존재). 그런데 `detseg-r_swin-b_internal.py`는 `seg_decoder`로 score map을 만든다. | `road_anomaly_heatmap.png` 등은 "랜덤 초기화 seg head의 RbA score + box 내부 +1.0"이다. box는 의미 있지만 heatmap 자체는 해석하면 안 된다. 로그의 score 범위(-18.8 ~ -13.1)도 이와 일치. |
| 2 | **VPHM(시간적 smoothing)은 배포 코드에서 사실상 구현되어 있지 않다.** | `mmdet/detseg_utils/losses.py:126` `VanishingPointGuidedCost` 하나뿐. `__call__` 안에서 정의되지 않은 변수(`pred_bboxes`, `gt_bboxes`, `self.iou_mode`)를 사용하고, 어떤 config·pipeline에서도 호출되지 않음. 비디오 입력 처리 코드도 없음. | VPHM은 재현/개선 대상에서 우선순위를 낮춰야 한다. |
| 3 | **UN(universal) query가 text 일부만 보도록 하는 mask가 실제로는 적용되지 않는 것으로 보인다.** | `models.py:1129`에서 `query_text_mask`를 만들지만, decoder layer의 text cross-attention(`mmdet/models/layers/transformer/grounding_dino_layers.py:111-116`)은 `text_attention_mask`만 사용. `query_text_mask`는 `**kwargs`로 흘러가 소비되는 곳이 없음 (repo 전체 grep 결과 사용처 0). | text prompt(ID class 목록)를 바꾸면 UN branch 결과도 바뀐다. query 수정 실험 시 반드시 고려. (실행 시 동작은 「확인 필요」) |
| 4 | **Road-overlap filter는 이미 뼈대가 구현되어 있지만, 기본값 off이고 road mask 품질이 문제다.** | `DetSegR.__init__(filter_by_road_overlap=False)`, `_get_road_overlap_mask`, `RoadOverlapRoIPooler`. road 영역에 label 0(=OoD 후보 자신)이 포함되고(`models.py:1588`), external 모드에서는 seg_pred가 전부 0(=road)이다. | "road-aware post-processing" 개선 주제의 출발점이 이미 코드에 있다 → 졸프 개선 주제로 매우 유리. |

---

## 1. DetSeg 코드 구조 지도

### Config — `configs/detseg/`

| 파일 | 역할 | 관련 논문 개념 |
|---|---|---|
| `detseg_swin-b_coco.py` | **Detection module 학습**. `type='DetSeg'`. MM-Grounding-DINO Swin-B에서 시작하고, backbone/encoder/decoder/language model은 `lr_mult=0`(고정). COCO box의 label을 전부 0으로 바꿔(`ReplacePrompt`) universal objectness를 학습. 배포 ckpt `detseg_swin-b_coco_*.pth`가 이 결과물이다. | UN query, open-world detection |
| `detseg-r_swin-b_external.py` | **DetSeg-R 평가(논문 재현용)**. `seg_decoder=None`. pipeline의 `GetAnomalyScoreMap`이 외부 방법(M2A, RbA 등)의 `.npy` score map을 불러온다. `box_score_thr=0.2`, `strong_objectness_box=1.0`, `filter_by_road_overlap=False`, `use_sam_refine=False`. | DetSeg-R |
| `detseg-r_swin-b_internal.py` | DetSeg-R + **내부 seg branch**(Mask2Former head, Cityscapes 19 class)로 score map을 직접 생성. 기본 dataset은 FS Static, evaluator는 `AnomalyMetricLoad`. **seg_decoder 가중치는 별도 학습이 필요**하다 (발견 #1). | DetSeg-R (내부 score) |
| `detseg-r_swin-b_cityscapes.py` | internal을 상속하고 `load_from=ckpt`만 지정. **내부 seg branch 학습용**(Cityscapes + COCO object paste). | DetSeg-R 학습 |
| `detseg-s_swin-b.py` | **DetSeg-S 평가**. `type='DetSegS'`, `sam_model_name='facebook/sam-vit-base'`, evaluator `BinaryMaskAnomalyIoUMetric`(IoU/F1). | DetSeg-S |
| `detseg_swin-b_obj365.py`, `_mix_data.py` | Objects365 / 혼합 데이터로 detection module을 학습하는 변형. | 학습 데이터 ablation (확인 필요) |

### Model — `mmdet/detseg_utils/models.py` (핵심)

| 클래스 | 줄 | 역할 | 관련 논문 개념 |
|---|---|---|---|
| `DetSeg(GroundingDINO)` | 836 | Grounding DINO에 **universal query 300개**를 추가한 detector. ID query 900개와 UN query 300개를 같은 decoder에 넣되 self-attention을 서로 차단. | open-world detection, ID/UN query |
| `GroundingDINOHeadWithUniversalObjectness` | 344 | 기존 text-contrastive 분류 head(ID)에 **1채널 objectness head(UN)** 를 추가. 두 결과를 합쳐 **IE-NMS** 적용. | ID/UN 분리, IE-NMS |
| `DetSegR(SamPromptMixin, DetSeg)` | 1217 | DetSeg box로 **기존 anomaly score map을 refine**(box 내부 +1.0). 선택적으로 road-overlap filter, SAM mask boost. | DetSeg-R |
| `DetSegS(SamPromptMixin, DetSeg)` | 1735 | DetSeg box를 **SAM box prompt**로 넣어 binary mask를 생성. threshold 탐색이 필요 없음. | DetSeg-S |
| `SamPromptMixin` | 256 | HuggingFace `SamModel`/`SamProcessor`를 lazy-load. state_dict에는 등록하지 않음. | SAM 연동 |
| `RoadOverlapRoIPooler` | 763 | road mask를 box 단위로 RoIAlign(avg) → box별 "road 위 비율"을 계산. | (논문 ablation) road-overlap filter |
| `Mask2FormerHeadAnomaly` | 148 | 내부 seg branch. Mask2Former + RbA loss. | DetSeg-R internal score |

### Dataset / Transform — `mmdet/detseg_utils/datasets.py`

| 클래스 | 줄 | 역할 | 관련 개념 |
|---|---|---|---|
| `ConcatPrompt` | 24 | text prompt 맨 앞에 **`'objects'`** 를 붙인다 → `('objects', 'road', ..., 'bicycle')`. 이 `objects` token이 UN branch의 anchor다. | UN query |
| `ReplacePrompt` | 61 | 학습 시 text를 Cityscapes 19 class로 교체하고 GT label을 모두 0으로 만든다. | UN objectness 학습 |
| `AddPrompt` | 47 | `objects` + 19 class를 직접 넣는 변형 (현재 config에서는 미사용) | — |
| `FSLostAndFoundDataset` 등 | 137, 75, 469, 536 | 평가 dataset. `get_data_info`에서 `text = METAINFO['classes']`(19 class), `custom_entities=True`로 설정(`:203-205`). | ID query 정의 |
| `GetAnomalyScoreMap` | 521 | `{data_path}/{이미지파일명}.npy`를 읽어 `results['anomaly_score_map']`에 넣는다. | DetSeg-R external |
| `PasteCocoObjects` | 370 | seg branch 학습용. COCO object를 Cityscapes에 붙여 pseudo-OoD를 만든다. | 내부 seg 학습 |

### Metric — `mmdet/detseg_utils/metrics.py`

| 클래스 | 줄 | 역할 |
|---|---|---|
| `AnomalyMetricLoad` | 736 | `anomaly_scores`(연속값)와 GT(0=ID, 1=OoD, 255=ignore)로 pixel-level **AUPRC / FPR@95TPR / AUROC** 계산 (DetSeg-R용) |
| `BinaryMaskAnomalyIoUMetric` | 637 | 이미 이진화된 `pred_sem_seg`로 **IoU / F1** 계산 (DetSeg-S용) |

### Inference / Evaluation

| 파일 | 역할 |
|---|---|
| `tools/test.py` | MMEngine `Runner.test()` 표준 진입점. DetSeg 전용 로직 없음. config의 `test_dataloader` → `model.predict` → `test_evaluator` 순서. |
| `tools/eval_box_recall.py` | GT mask에서 box를 만들어 **OoD box recall**을 측정 (`--min-area` 옵션 있음). **작은 장애물 분석에 바로 쓸 수 있다.** |
| `tools/eval_detseg_r_official.py` | README 표(M2A/RbA/UNO… × 3 dataset)를 재생성. 외부 score map은 `other_score_results/...`에 직접 준비해야 함. |
| `src/run_detseg_single_image.py` (우리 repo) | `SetDetSegPrompt`로 text를 넣고 internal config로 단일 이미지 추론. **발견 #1 때문에 heatmap은 무의미. box만 유효.** |

---

## 2. 전체 pipeline (DetSeg-R external 기준, 단계별)

```text
이미지
 │ [Data pipeline: configs/detseg/detseg-r_swin-b_external.py]
 ├─ LoadImageFromFile
 ├─ FixScaleResize(scale=(800,1333), keep_ratio)   ← 2048×1024 → 약 1333×666 (작은 물체가 더 작아짐)
 ├─ LoadAnnotations(seg)                           ← GT anomaly mask
 ├─ (dataset.get_data_info) text = 19 Cityscapes class, custom_entities=True
 ├─ ConcatPrompt        → text = ('objects', 'road', ..., 'bicycle')   [20 entity]
 ├─ GetAnomalyScoreMap  → metainfo['anomaly_score_map'] = 외부 .npy
 └─ PackDetInputs
 │ [DetDataPreprocessor] 정규화, BGR→RGB, padding
 ▼
DetSegR.predict  (models.py:1410)
 ├─ ① get_tokens_positive_and_prompts  → caption "objects. road. ... bicycle."
 │      token_positive_map: entity 1='objects', 2='road', ...   (label = entity-1 → 'objects'=label 0)
 ├─ ② extract_feat: Swin-B → ChannelMapper neck (detection feature)
 ├─ ③ anomaly score map 준비 (1465-1502)
 │      external이면 metainfo의 map을 원본 크기로 보간
 │      아니면 seg_decoder로 -Σ tanh(logits[:19]) (RbA score)
 ├─ ④ BERT → text_feat_map → forward_transformer (1017)
 │      'objects' token embedding += universal_embedding
 │      encoder: image–text fusion (6 layer)
 │      pre_decoder (1044): ID용 top-900 / UN용 top-300 proposal을 따로 선택
 │      decoder (6 layer): [900 ID query | 300 UN query], 서로 self-attn 차단
 ├─ ⑤ bbox_head.predict → predict_by_feat (482)
 │      ID: text 유사도 → entity별 score → top-300  (label 0~19)
 │      UN: objectness sigmoid → top-300            (label 0)
 │      concat → _ie_nms (518)   ← IE-NMS
 ├─ ⑥ DetSeg-R 후처리 (1582-1624)
 │      (옵션) road-overlap filter
 │      label == 0 만 남김  (= OoD 후보)
 │      score > box_score_thr(0.2)
 │      _boost_anomaly_score: box 내부 pixel score += 1.0   (옵션: SAM mask 내부만)
 └─ ⑦ 출력: pred_instances(OoD box), anomaly_scores(refined map), pred_masks
 ▼
AnomalyMetricLoad → AUPRC / FPR95 / AUROC
```

**DetSeg-S**는 ①~⑤가 같고(`DetSeg.predict` 재사용), ⑥이 다르다: `label==0 & score>0.2`(하드코딩, `models.py:961`) → 남은 box를 SAM에 prompt → mask 합집합을 `pred_sem_seg`로 출력 → IoU/F1.

---

## 3. 논문 개념 ↔ 코드 위치 매핑표

| 논문 개념 | 코드 위치 | 핵심 함수/클래스 | 역할 | 수정 가능성 |
|---|---|---|---|---|
| ID query | `models.py:987` (`query_embedding`, 900개, Grounding DINO 원본), `:1068` ID용 top-k, `datasets.py:203` text 정의 | `DetSeg._init_layers`, `pre_decoder`, `GroundingDINOHead._predict_by_feat_single` | text prompt(19 class)와의 contrastive 유사도로 **알려진 class box**를 찾는다 | **높음(코드), 주의(효과)**: class 목록은 dataset METAINFO / 우리 스크립트의 `SetDetSegPrompt`에서 바꿀 수 있음. 단 `_ie_nms`의 threshold 표가 19 class로 하드코딩되어 있어 함께 수정해야 함 |
| UN query (universal) | `models.py:1005-1015` (`universal_query_embedding` 300개, `universal_embedding`, `universal_memory_trans_*`), `:1076-1093` UN용 top-k, `:353` `objectness_branches` | `DetSeg.pre_decoder`, `gen_universal_encoder_output_proposals`, `_predict_universal_by_feat_single` | class와 무관한 **"물체다/아니다" objectness**로 box 후보를 만든다. 모든 출력 label = 0 | 중간: 개수(300)는 코드 상수. 재학습 없이 바꿀 수 있는 것은 top-k, threshold, 후처리뿐 |
| Open-world detection | `mmdet/models/detectors/grounding_dino.py` (원본) + `DetSeg` | `DetSeg.forward_transformer`, `GroundingDINOHeadWithUniversalObjectness.forward` | Grounding DINO 위에 UN branch를 덧붙인 구조 | 백본 수정은 재학습 필요 → 비권장 |
| IE-NMS | `models.py:518-541` | `GroundingDINOHeadWithUniversalObjectness._ie_nms` | ID box 중 "확실한" 것의 score를 ×2 해서 NMS 경쟁에서 이기게 함 → ID와 겹친 UN box 제거 | **매우 높음**: 20줄 남짓의 순수 후처리, 재학습 불필요 |
| DetSeg-R | `models.py:1217-1632` | `DetSegR.predict`, `_boost_anomaly_score` | 외부/내부 score map에서 OoD box 내부 pixel에 +1.0 → box 밖 FP를 상대적으로 낮춤 | 높음: 후처리만 수정하면 됨 |
| DetSeg-S | `models.py:1735-1769`, `:256-335` | `DetSegS.predict`, `SamPromptMixin.sam_masks_from_boxes` | OoD box → SAM → binary mask (threshold 없음) | 높음: SAM 출력 선택·후처리 수정 가능 |
| VPHM | `losses.py:126-162` | `VanishingPointGuidedCost` | (의도) 소실점 기준 각도를 matching cost로 사용해 프레임 간 box 연결 | **미완성**: 실행 불가 코드, 호출처 없음 (발견 #2) |
| Anomaly score map | external: `datasets.py:521` `GetAnomalyScoreMap` → `models.py:1465-1487` / internal: `models.py:1489-1502` | `GetAnomalyScoreMap`, `DetSegR.predict`, `Mask2FormerHeadAnomaly` | DetSeg-R의 base map. 모델이 만드는 게 아니라 **외부 방법이 만든 map을 받아 보정**하는 것이 논문 재현 경로 | 결합 방식(+1.0)은 쉽게 수정 가능 |
| Road-overlap filter (부가) | `models.py:763`, `:1388-1408`, `:1586-1614` | `RoadOverlapRoIPooler`, `_get_road_overlap_mask` | box의 road 위 평균 비율 > 0.4 또는 두 모서리가 road 위면 유지 | **높음**: 개선 후보 1의 출발점 |

---

## 4. 핵심 함수별 설명

### 4.1 `ConcatPrompt.transform`

위치:
```text
external/DetSeg-official/mmdet/detseg_utils/datasets.py:24-43
```
- 역할: text prompt 앞에 `'objects'`를 붙인다.
- 입력: `results['text']` (tuple 또는 str)
- 출력: `('objects', 'road', 'sidewalk', ...)`
- 논문 개념: UN query가 붙잡을 "범용 물체" token
- 코드 흐름:
  1. tuple이면 `('objects',) + text`
  2. str이면 `'objects. ' + text`, `tokens_positive`를 글자 9개(`"objects. "`)만큼 이동
- 수정 가능성: `objects` 대신 다른 단어(예: `obstacle`)로 바꿀 수 있으나, **학습 때도 `objects`였으므로** 바꾸면 성능이 떨어질 가능성이 높다.
- 주의점: 이 transform은 반드시 `text`를 만드는 단계 **뒤에** 와야 한다. 단일 이미지에서 `KeyError: 'text'`가 났던 이유.

### 4.2 `DetSeg.forward_transformer` / `DetSeg.pre_decoder`

위치:
```text
models.py:1017-1042 (forward_transformer), 1044-1152 (pre_decoder)
```
- 역할: ID query 900개와 UN query 300개를 **하나의 decoder에서 독립적으로** 돌리도록 입력을 준비
- 입력: Swin feature, BERT text feature
- 출력: decoder 입력(query, reference points, attention mask)
- 논문 개념: ID query / UN query 이중 구조
- 코드 흐름:
  1. (`:1026`) text embedding의 1번 token(`objects`)에 학습된 `universal_embedding`을 더함
  2. encoder가 image–text를 융합
  3. (`:1055-1074`) **ID proposal**: encoder 출력의 text 유사도 max로 top-900 위치 선택
  4. (`:1076-1093`) **UN proposal**: 별도 projection(`universal_memory_trans_fc/norm`) → `objectness_branches[6]` 점수로 top-300 위치 선택. box 좌표는 ID와 같은 reg branch 출력 사용
  5. (`:1095`) query = `[query_embedding(900) ; universal_query_embedding(300)]`
  6. (`:1124-1125`) self-attention mask: ID↔UN 서로 못 봄 → 두 집합이 독립
  7. (`:1129`) UN query는 text token 3번 이후를 못 보게 하려는 `query_text_mask` 생성 → **실제로는 미적용으로 보임** (발견 #3)
- 수정 가능성: 낮음(재학습 영역). top-k 개수(300)는 상수로 바꿀 수는 있다.
- 주의점: UN proposal 선택은 **encoder 단계 top-300**이 1차 관문이다. 작은 물체가 여기서 빠지면 후처리로 되살릴 수 없다.

### 4.3 `GroundingDINOHeadWithUniversalObjectness.predict_by_feat`

위치:
```text
models.py:482-516
```
- 역할: decoder 마지막 layer 출력을 box로 변환하고 ID/UN 결과를 합친다
- 입력: cls score(ID), objectness score(UN), box
- 출력: IE-NMS를 거친 `InstanceData(bboxes, scores, labels)`
- 코드 흐름:
  1. **ID**: `GroundingDINOHead._predict_by_feat_single` (`grounding_dino_head.py:385`) → `convert_grounding_to_cls_scores`(`atss_vlfusion_head.py:31`)가 token 유사도를 entity별 평균으로 바꿈 → (900×20)에서 top-300. **label = entity index** (0='objects', 1='road', ..., 19='bicycle')
  2. **UN**: `_predict_universal_by_feat_single` (`:543`) → sigmoid objectness top-300, **label = 0**
  3. concat (최대 600개) → `_ie_nms`
- 주의점 (확인 필요): ID branch도 `'objects'` entity에 대해 label 0을 낼 수 있다. 즉 최종 "OoD 후보(label 0)"에는 **UN branch 결과 + ID branch의 'objects' 응답**이 섞여 있다. 실제 비율은 디버그 출력으로 확인해야 함.

### 4.4 `_ie_nms` (IE-NMS) ★

위치:
```text
models.py:518-541
```
```python
thres = ones                                      # 기본 임계값 1.0 (= 강화 불가)
id_thresholds = [0.2]*11 + [0.5]*8                # label 1~11(road..sky)=0.2, 12~19(person..bicycle)=0.5
id_mask = (labels != 0) & (scores > thres)        # "확실한 ID box"
nms_scores[id_mask] *= 2                          # 점수 2배로 NMS 우선권
batched_nms(..., iou_threshold=0.5, class_agnostic=True)
scores[id_mask[keep]] /= 2                        # 출력 score는 원래대로
```
- 역할: ID box와 UN box가 겹치면 **확실한 ID box가 이기도록** 해서 UN 쪽의 "사실은 알려진 물체" 후보를 제거
- 일반 NMS와 차이:
  - 일반 NMS: 같은 class끼리, score 순으로 IoU>thr 제거
  - IE-NMS: **class 구분 없이**(`class_agnostic=True`) 한 번에 NMS하되, ID box(class별 임계값 초과)의 순위를 ×2로 올림
- ID/UN box가 겹칠 때 기준: IoU > 0.5이고 `2·s_ID > s_UN`이면 UN box 제거. 반대로 ID box가 임계값 미달이면 일반 score 경쟁
- IoU threshold: `:533` 하드코딩 `0.5`. score 기준: `:522` 하드코딩 표. **config로 노출되어 있지 않음**
- 이후 단계: label 0만 남기므로(DetSeg-R `:1616`, DetSeg/S `:961-963`), NMS에서 살아남은 ID box는 결국 버려지고 **"ID box에 먹히지 않은 label-0 box"만 OoD로 남는다**

**잘못 제거될 가능성 (약점):**
1. **OoD 물체가 ID class로 오인되는 경우**: 라바콘 → `traffic sign`/`pole`, 낙하 박스 → `car`의 일부, 동물 → `person`. 특히 label 1~11은 임계값이 **0.2로 낮다**(pole, traffic light, traffic sign 포함). 같은 위치(IoU>0.5)에 score 0.2짜리 `pole` box만 있어도 ×2=0.4가 되어 objectness 0.35인 UN box를 제거한다.
2. **IoU만 본다**: 작은 낙하물이 큰 ID box(예: `car`, `road`) 안에 완전히 포함돼도 IoU는 작으므로 제거되지 않는다 → 이 경우는 오히려 안전. 반대로 크기가 비슷하게 겹치는 경우에만 제거된다.
3. **하드코딩된 19-class 표**: prompt 순서/개수를 바꾸면 임계값이 엉뚱한 class에 붙는다. 20번째 이후 class는 `thres=1`이 되어 절대 강화되지 않는다.

**개선 가능성:**
- 작은 장애물 보존: `area(UN box)`가 작으면 제거 조건을 강화(예: IoU>0.7일 때만 제거)
- 도로 내부 보존: UN box 하단 중앙점이 road mask 안이면 ID에 의한 제거 면제
- 높은 objectness 보존: `s_UN > τ_high`(예: 0.6)인 UN box는 ID와 겹쳐도 유지
- stuff class(road, building, sky, vegetation, terrain, wall)는 OoD를 억제할 이유가 거의 없으므로 강화 대상에서 제외

### 4.5 `DetSeg.predict`

위치:
```text
models.py:843-965
```
- 역할: DetSeg / DetSeg-S의 공통 box 예측
- 코드 흐름: prompt 토큰화 → feature → transformer → head → label name 부여 → **`keep = (score > 0.2) & (label == 0)` (`:961-963`, 하드코딩)**
- 수정 가능성: 높음. DetSeg-S의 box threshold를 바꾸려면 **여기를 고쳐야 한다** (config 파라미터 없음)
- 주의점: DetSeg-R은 이 함수를 쓰지 않고 자체 `predict`를 가지므로, 공통 개선은 `_ie_nms`나 별도 helper로 넣는 것이 깔끔하다

### 4.6 `DetSegR.predict` ★

위치:
```text
models.py:1410-1632
```
- 역할: OoD box로 pixel anomaly score map을 refine
- 입력: 이미지, text, (선택) `anomaly_score_map`
- 출력: `pred_instances`(OoD box), `anomaly_scores`(refined map), `pred_masks`
- 논문 개념: DetSeg-R
- 코드 흐름:
  1. (`:1465-1487`) 모든 샘플에 외부 map이 있으면 사용. 이때 `seg_preds = 0`(전부 road로 간주)
  2. (`:1488-1502`) 없으면 `seg_decoder` → **RbA score** `-Σ_c tanh(logit_c)`. seg_decoder가 없으면 `RuntimeError`
  3. (`:1504-1558`) detection (4.2~4.4와 동일)
  4. (`:1586-1609`) road mask = (label∈{0,1,2}이고 score>0.2인 box 영역) ∩ (seg_pred ∈ {road, sidewalk})
  5. (`:1611-1614`) `filter_by_road_overlap=True`일 때만 road 위 box만 유지
  6. (`:1616`) label 0만 유지 → (`:1619`) `score > box_score_thr`(0.2)
  7. (`:1622`) `_boost_anomaly_score`: 남은 box 내부 pixel에 `+strong_objectness_box`(1.0)
- 수정 가능성: **가장 높음**. 6~7단계가 순수 후처리이고 config 파라미터로 이미 일부 노출됨
- 주의점:
  - **+1.0은 절댓값**이다. M2A(0~1 확률)에는 강하게, RbA(넓은 범위의 logit)에는 약하게 작용 → base method마다 효과가 다를 수 있다 (README 표에서 방법별 개선 폭이 다른 이유일 수 있음, 확인 필요)
  - box 내부 **배경 pixel도 같이** 올라간다 → box 모양의 FP가 생긴다. pixel AP를 깎는 요인
  - `road_label_ids = arange(3)` → **label 0(OoD 후보 자신)도 road 영역으로 친다**. 그래서 OoD box는 자기 자신 때문에 road 위로 판정될 수 있다 → filter가 사실상 무력화될 수 있다 (의도 확인 필요)

### 4.7 `DetSegR._boost_anomaly_score`

위치:
```text
models.py:1363-1386
```
- 기본: box 영역 합집합에 `+box_boost`
- `use_sam_refine=True`: SAM mask 합집합에만 `+box_boost`
- 개선 포인트: 상수 boost → `boost × objectness`, `boost × normalized(base score)` 같은 **soft fusion**으로 바꾸기 쉽다

### 4.8 `DetSegR._get_road_overlap_mask`

위치:
```text
models.py:1388-1408
```
- box 내부 road mask 평균 > 0.4, **또는** box의 (x1,y1)·(x2,y2) 두 모서리가 모두 road 위 → 유지
- 주의점: (x1,y1)은 **좌상단** 모서리이다. 도로 위 물체의 좌상단은 보통 도로 밖(배경)에 있으므로 이 조건은 약하다. 바닥 중앙점(`(x1+x2)/2, y2`)이 더 자연스러운 기준이다

### 4.9 `DetSegS.predict` + `SamPromptMixin`

위치:
```text
models.py:1749-1769 (DetSegS.predict)
models.py:256-335   (SamPromptMixin)
```
- 역할: OoD box → SAM → binary mask
- 코드 흐름:
  1. `DetSeg.predict` → label 0 & score>0.2 box
  2. `sam_masks_from_boxes` → `sam_predict_hf`:
     - `SamModel.from_pretrained('facebook/sam-vit-base')` (HuggingFace, 첫 호출 시 다운로드)
     - **원본 이미지 파일을 다시 읽어서**(`Image.open(img_path)`) SAM image embedding 계산
     - box를 `input_boxes`로 전달 → `post_process_masks`로 원본 크기 복원
  3. `masks[0][:, 0]` → **box마다 SAM의 3개 multimask 후보 중 첫 번째만 사용**. `iou_scores`는 버린다 (확인됨: transformers 4.46.3 `SamModel.forward`의 `multimask_output` 기본값 `True`)
  4. 모든 mask OR → `pred_sem_seg`
- 개선 포인트:
  - 3개 후보 중 SAM `iou_scores`가 가장 높거나 **anomaly heatmap과 가장 일치하는** mask를 선택
  - mask를 box 안으로 clip (SAM이 box 밖으로 번지는 경우)
  - mask 면적 / box 면적 비율이 비정상이면 제거
  - road mask와 교집합
- 주의점: SAM은 box 하나하나에 mask를 반드시 낸다 → **box FP는 그대로 mask FP**가 된다. DetSeg-S 성능은 box 품질에 직결된다

### 4.10 `GetAnomalyScoreMap` / `AnomalyMetricLoad`

위치:
```text
datasets.py:521-531, metrics.py:736-824
```
- `GetAnomalyScoreMap(data_path)`: `{data_path}/{basename(img_path)}.npy` 로드 (예: `other_score_results/score_results_m2a/FS_LostFound/xxx.png.npy`)
- `other_score_results/score_results_m2a/FS_LostFound`: **Mask2Anomaly(M2A)가 FS Lost&Found에 대해 미리 계산한 pixel score map 폴더**. repo에 포함되어 있지 않으며 로컬에도 없음 → 직접 생성하거나 저자에게 받아야 함 (확인 필요)
- `AnomalyMetricLoad`: 이름의 "Load"는 "미리 계산된(load된) score map을 평가한다"는 뜻. GT==1 pixel을 positive, GT==0을 negative, 255 무시 → `roc_curve`, `average_precision_score`, `fpr_at_95_tpr`

**결론: DetSeg-R external은 "외부 score map 필수" 구조이다.** internal은 seg_decoder 학습이 선행되어야 한다.

### 4.11 `VanishingPointGuidedCost` (VPHM)

위치:
```text
losses.py:126-162
```
- 의도: 소실점 `vp`(기본 `[512, 1024]`)에서 두 box 중심으로 그은 선의 **각도 차이**를 cost로 써서, 연속 프레임의 box를 Hungarian matching으로 연결
- 실제 상태:
  - `calc_angle`만 의미 있게 작성됨
  - `__call__`은 IoU cost 코드를 복사한 흔적이고 정의되지 않은 변수를 사용 → 호출하면 `NameError`
  - 어떤 config, transform, model에서도 사용하지 않음
- 입력/출력(의도): 이전 프레임 `InstanceData`, 현재 프레임 `InstanceData` → cost matrix
- 결론: 배포 코드로는 VPHM 재현 불가. 구현하려면 비디오 dataset, frame loop, matching, smoothing을 모두 새로 작성해야 한다

---

## 5. 질문별 답변

### 5.1 ID/UN query와 위험 객체 query 추가

- **ID query 정의 위치**: 학습 파라미터는 `query_embedding`(`models.py:987`). "어떤 class가 ID인가"는 **text prompt**가 결정한다
  - 평가 dataset: `FSLostAndFoundDataset.METAINFO['classes']` → `get_data_info`의 `data_info['text']` (`datasets.py:157, 203`)
  - 단일 이미지: 우리 스크립트의 `SetDetSegPrompt` (`src/run_detseg_single_image.py:79-105`)
- **UN query 정의 위치**: `universal_query_embedding`(`:1006`) + `ConcatPrompt`의 `'objects'` token + `universal_embedding`
- **text prompt 생성**: `('objects', 19 class)` → `get_tokens_positive_and_prompts`가 `"objects. road. ... bicycle."` caption과 entity→token map을 만든다
- **known/unknown 구분**: 오직 **label 값**이다. label 0 = unknown 후보, label 1~19 = known. 구분은 IE-NMS 이후 `labels == 0` 필터에서 일어난다

**위험 객체(라바콘, 타이어, 박스, 동물 등) query를 추가할 수 있는가?**

추가는 가능하지만, **그냥 prompt에 넣으면 역효과**다.

- prompt에 들어간 class는 전부 "ID(known)"로 처리된다 → 라바콘을 prompt에 넣으면 라바콘 box가 label≠0이 되어 **IE-NMS에서 OoD 후보를 억제하고, 최종 필터에서 버려진다**. 즉 라바콘을 "못 찾게" 된다
- 제대로 하려면 **세 번째 그룹 "known hazard"** 를 만들어야 한다:

| 수정 파일 / 함수 | 수정 내용 |
|---|---|
| prompt 정의 (`datasets.py` METAINFO 또는 `SetDetSegPrompt`) | 19 class 뒤에 hazard class 추가 (예: `'traffic cone', 'tire', 'cardboard box', 'animal'`) |
| `_ie_nms` (`models.py:518`) | `id_thresholds`를 길이에 맞게 확장. hazard label은 **강화 대상에서 제외**(UN box를 억제하지 않도록) |
| `DetSeg.predict` (`:961`), `DetSegR.predict` (`:1616`) | `labels == 0` → `labels == 0 or label ∈ hazard_ids`로 변경해 hazard box도 OoD로 유지 |

- 주의 (중요):
  - **UN branch는 항상 19-class prompt로 학습됐다**(`ReplacePrompt`). 발견 #3 때문에 prompt를 바꾸면 UN 출력도 변할 수 있다 → 반드시 baseline(19 class)과 비교 실험해야 한다
  - 반대 방향의 개선도 가능하다: 도로 FP의 흔한 원인(`manhole`, `road marking`, `puddle`, `shadow`)을 **ID로 추가해 억제**

### 5.2 Open-world detection branch와 threshold

| 질문 | 답 |
|---|---|
| Grounding DINO 구현 위치 | `mmdet/models/detectors/grounding_dino.py`(원본) → `DetSeg`가 상속 (`models.py:836`) |
| box proposal 생성 | 1차: `pre_decoder`의 encoder top-k (ID 900 / UN 300). 최종 box: decoder 6 layer refine 후 `predict_by_feat` |
| detection score 계산 | ID: `cls_branches`(text contrastive) → `convert_grounding_to_cls_scores`. UN: `objectness_branches` → sigmoid (`:554`) |
| score threshold 적용 위치 | (a) IE-NMS 강화 임계값 `:522` (b) DetSeg/S 최종 `0.2` `:961` (c) DetSeg-R 최종 `box_score_thr` `:1619` (d) road mask용 `0.2` `:1587` (e) 우리 스크립트 `--score-thr`는 **시각화만** 거른다 |
| threshold 변경 영향 | (b)(c)를 올리면 OoD box 수가 줄어든다 → DetSeg-R은 boost 영역 감소(FP↓, recall↓), DetSeg-S는 mask 수 감소. 우리 스크립트의 0.20/0.30/0.50 비교는 모델 내부 0.2 **이후**의 시각화 필터라서, **0.2 미만 값은 효과가 없다** |

**개선 관점:**
- **Adaptive threshold**: 가능. 이미지별 objectness 분포(예: 상위 k개 평균 대비 비율), box 면적별 threshold, 위치별 threshold. 모두 `:1619`(R) / `:961`(S)에서 구현 가능
- **도로 내부 box re-ranking**: 가능. `road_overlap_pooler`가 이미 있어 box별 road 비율을 쉽게 얻는다. hard filter 대신 `score' = score × (α + (1-α)·road_ratio)`로 바꾼 뒤 threshold 적용
- **작은 낙하물이 score 때문에 제거될 가능성**: **높다고 판단 (실험으로 확인 필요)**. 근거:
  1. 입력 resize(2048→약 1333 폭)로 원래 작은 물체가 더 작아진다
  2. 작은 물체는 objectness가 낮게 나오는 경향이 일반적이다 (DETR 계열 공통 약점)
  3. encoder top-300 단계에서 이미 탈락할 수 있다 (후처리로 복구 불가)
  4. 최종 0.2 threshold가 면적과 무관하다
  → `tools/eval_box_recall.py --min-area`로 **면적 구간별 recall**을 재면 바로 정량화할 수 있다

### 5.3 DetSeg-R vs DetSeg-S

| 항목 | DetSeg-R | DetSeg-S |
|---|---|---|
| class | `DetSegR` (`:1217`) | `DetSegS` (`:1735`) |
| config | `detseg-r_swin-b_external.py` / `_internal.py` | `detseg-s_swin-b.py` |
| config 차이 | seg_decoder, road_overlap_pooler, box_score_thr, boost, `GetAnomalyScoreMap`, `AnomalyMetricLoad` | sam_model_name, `BinaryMaskAnomalyIoUMetric`, score map 불필요 |
| forward 흐름 | 자체 `predict`: score map 준비 → detection → road/label/score 필터 → box boost | `DetSeg.predict` 재사용 → SAM → mask OR |
| score map 사용 | base map + box 내부 +1.0 | 사용 안 함 |
| SAM 사용 | 옵션(`use_sam_refine`) | 필수 |
| 평가 지표 | AUROC / AP / FPR95 (pixel) | IoU / F1 (binary mask) |
| 평가 준비물 | dataset + **외부 score map**(M2A 등) 또는 **학습된 seg branch** | dataset + SAM 가중치(자동 다운로드) |
| 코드 수정 난이도 | 하 (후처리 함수 하나) | 하~중 (SAM 출력 처리) |
| 실험 난이도 | **중~상**: score map 확보가 병목 | **하**: dataset만 있으면 바로 평가 |

**권장: box 단계(IE-NMS, threshold, road-aware)를 개선하고, 평가는 DetSeg-S + box recall로 먼저 한다.**
- box 단계 개선은 R/S 공통이므로 한 번 고치면 양쪽에 적용된다
- DetSeg-S는 외부 score map 없이 FS Lost&Found(100장)/RoadAnomaly(60장)만 있으면 평가할 수 있다
- DetSeg-R은 score map을 확보한 뒤(M2A 생성 또는 seg branch 학습) 2단계로 확장한다

### 5.4 SAM / mask refinement

| 질문 | 답 |
|---|---|
| SAM 로드 위치 | `SamPromptMixin.sam` (`:267`) — `transformers.SamModel.from_pretrained(sam_model_name)`. 첫 사용 시 lazy-load, state_dict에 미포함 |
| box prompt 전달 | `sam_predict_hf` (`:299`): `processor(raw_image, input_boxes=[boxes.tolist()])` → image embedding 재사용 → `sam_model(**inputs)` |
| 후처리 | `post_process_masks`(원본 크기 복원, 이진화) → 각 box의 첫 번째 multimask만 선택 → 크기 보간 → OR 합집합 |
| SAM mask가 너무 넓을 때 | box clip, 3개 후보 중 가장 작은/IoU 높은 mask 선택, mask/box 면적비 필터 → **가능 (난이도 하)** |
| 도로 밖 mask 제거 | road mask와 교집합 → **가능** (road mask 확보가 전제) |
| heatmap + SAM 결합 | DetSeg-R `use_sam_refine` 경로가 이미 있다. `objectness[mask] = boost` → `boost × mask × norm(heatmap)` 같은 soft fusion, 또는 heatmap 평균이 낮은 mask 제거 → **가능 (난이도 중)** |

---

## 6. 개선 후보 리스트와 우선순위

| 순위 | 후보 | 난이도 | 적합도 | 재학습 |
|---|---|---|---|---|
| 1 | Road-aware unknown box re-ranking | 중 | 상 | 불필요 |
| 2 | IE-NMS 개선 (보호 규칙 + 설정화) | 하 | 상 | 불필요 |
| 3 | 작은 도로 낙하물 보존 (면적 적응 threshold + 고해상도 추론) | 하~중 | 상 | 불필요 |
| 4 | Heatmap + SAM mask refinement | 중 | 중상 | 불필요 |
| 5 | Query set 개선 (hazard / FP-negative prompt) | 하(코드)·중(실험) | 중 | 불필요 |
| — | VPHM 구현 | 상 | 하 | 불필요하나 비디오 데이터·구현량 큼 |

### 개선 후보 1: Road-aware unknown box re-ranking

**아이디어**
도로(주행 가능 영역) 위 또는 ego-lane 근처의 unknown box는 점수를 올리고, 도로 밖(건물, 하늘, 인도 안쪽)의 unknown box는 점수를 내린다. hard filter가 아니라 **soft re-ranking**을 적용한다.

**왜 필요한가**
- 자율주행에서는 주행 가능 영역 위의 물체가 직접적인 위험이다
- Lost&Found / RoadObstacle 계열 GT는 대부분 도로 위에 있다 → 도로 밖 OoD box는 FP일 가능성이 높다
- 코드에 이미 road filter 뼈대가 있지만 (a) 기본 off, (b) road 정의에 label 0이 포함, (c) external 모드에서 seg가 없어 road mask가 의미 없음, (d) 좌상단 모서리 기준이라는 약점이 있다

**수정할 코드 위치**
- `models.py` `DetSegR.predict` `:1586-1620` (road mask 생성, 필터)
- `models.py` `DetSegR._get_road_overlap_mask` `:1388-1408`
- DetSeg-S에도 적용하려면 `DetSeg.predict` `:961` 직전에 같은 helper 호출
- 새 helper 함수 권장: `_road_aware_rescore(pred_instances, road_mask)`

**구현 방법**
1. road mask 확보 (아래 "필요한 추가 입력" 중 하나)
2. box별 road 정보 계산: `road_ratio` = box 내부 road 비율 (기존 `RoadOverlapRoIPooler` 재사용), `foot_on_road` = 바닥 중앙점 `((x1+x2)/2, y2)`가 road 위인지
3. 재점수: `s' = s × (α + (1−α)·max(road_ratio, foot_on_road))` (α≈0.3~0.5)
4. threshold는 `s'`에 적용
5. ablation: hard filter(기존) vs soft re-rank vs off

**필요한 추가 입력**
- **road mask가 필요하다.** 선택지는 세 가지:
  - (a) **Grounding DINO 자체의 `road` box**(label 1): 추가 모델 없이 즉시 사용 가능. 단 box라서 거칠다
  - (b) **사전학습 Cityscapes segmentation 모델**(mmseg 설치되어 있음, 예: SegFormer/Mask2Former Cityscapes ckpt): road/sidewalk class mask. 가장 정확, 추론 비용 추가
  - (c) **위치 prior**: 이미지 하단 중앙의 사다리꼴(소실점 기반). 가장 단순, baseline으로 좋음
- 권장: (c)와 (a)로 먼저 효과를 확인하고, (b)를 최종안으로 사용

**난이도**: 중
**졸업프로젝트 적합도**: 상
**예상 위험**
- RoadAnomaly에는 도로 밖 이상물체(들판의 동물 등)가 있다 → soft 방식이어야 하고, dataset별 결과를 나눠 보고해야 한다
- 물체가 도로를 가리므로 seg 모델은 물체 위치를 "road 아님"으로 낼 수 있다 → box 내부 비율보다 **바닥 중앙점/box 주변 띠 영역** 기준이 안전하다

### 개선 후보 2: IE-NMS 개선

**아이디어**
IE-NMS가 UN box를 제거하는 조건에 **보호 규칙**을 추가하고, 하드코딩된 값을 config로 노출한다.

**왜 필요한가**
- 임계값 0.2인 label 1~11에 `pole/traffic light/traffic sign`이 포함되어, 라바콘·표지판 형태의 OoD가 약한 ID box에 쉽게 억제될 수 있다
- IoU만으로 판단하고 UN 쪽 신뢰도는 보지 않는다
- 19 class 하드코딩이라 query 실험(후보 5)을 하려면 어차피 고쳐야 한다

**수정할 코드 위치**
- `models.py` `GroundingDINOHeadWithUniversalObjectness._ie_nms` `:518-541`
- (선택) `predict_by_feat` `:482`에서 `_ie_nms` 호출 시 설정 전달
- config: `bbox_head=dict(..., ie_nms_cfg=dict(...))` 추가

**구현 방법**
1. `id_thresholds`, `iou_threshold`, `boost(=2)`를 `__init__` 인자 / config로 이동
2. 보호 규칙 (각각 on/off 가능하게):
   - **R1 high-objectness**: `s_UN > τ_keep`인 UN box는 ID와 겹쳐도 유지
   - **R2 small-object**: 면적 < A_small인 UN box는 ID와 IoU>0.7일 때만 제거
   - **R3 stuff-exempt**: road/sky/building/vegetation/terrain/wall/sidewalk는 강화하지 않음 (OoD를 억제할 이유가 없는 class)
3. 구현 팁: `batched_nms`를 두 번 쓰면 간단하다 — (i) UN 내부 NMS, (ii) ID↔UN 억제는 `bbox_overlaps` 행렬로 직접 계산하고 보호 규칙 적용
4. 원래 IE-NMS / 일반 NMS / IE-NMS+R1/R2/R3 ablation 표 작성

**필요한 추가 입력**: 없음
**난이도**: 하
**졸업프로젝트 적합도**: 상 (논문 핵심 모듈의 직접 개선이라 보고서에서 설득력이 크다)
**예상 위험**
- 보호를 늘리면 ID 물체(차, 사람)가 OoD로 새어 나와 FP가 증가 → FPR95가 나빠질 수 있다. recall–FP trade-off 곡선으로 보고해야 한다

### 개선 후보 3: 작은 도로 낙하물 보존

**아이디어**
작은 box에는 더 낮은 threshold를 적용하고(면적 적응 threshold), 필요하면 고해상도/타일 추론으로 작은 물체의 objectness를 높인다.

**왜 필요한가**
- 2048×1024 입력이 약 1333×666으로 줄어든다 (`FixScaleResize(800,1333)`)
- Lost&Found 장애물은 매우 작다 (수십 px)
- 최종 threshold 0.2가 면적과 무관하다
- 실제 도로 낙하물은 "작고, 멀고, 도로 위"인 경우가 대부분이다

**수정할 코드 위치**
- `models.py` `DetSegR.predict` `:1619` / `DetSeg.predict` `:961` (threshold)
- `_predict_universal_by_feat_single` `:550` (top-k를 300 → 늘리기, 실험용)
- config `test_pipeline`의 `FixScaleResize scale` (고해상도 추론, 코드 수정 없음)
- 평가: `tools/eval_box_recall.py --min-area` (면적별 recall)

**구현 방법**
1. **먼저 진단**: box recall을 면적 구간(예: <32², 32²~96², >96²)별로 측정 → 작은 물체에서 recall이 떨어지는지 확인 (이것만으로도 보고서 분석 섹션이 된다)
2. `thr(area) = thr_small if area < A_s else thr_base` 또는 연속 함수
3. 입력 scale 1333 → 2048(원본) 비교 (Swin window 연산이라 메모리 확인 필요. RTX 4070 Ti 12GB에서 가능 여부 확인 필요)
4. (선택) 이미지 하단 절반을 crop·확대해 2-pass 추론 후 box 합치기

**필요한 추가 입력**: 없음 (GT mask로 box recall 계산)
**난이도**: 하~중
**졸업프로젝트 적합도**: 상 (진단 → 원인 → 개선 흐름이 깔끔하다)
**예상 위험**
- threshold를 낮추면 작은 FP(돌, 노면 표시)가 증가 → 후보 1(road-aware)과 결합해야 효과적이다
- encoder top-300에서 이미 탈락한 물체는 threshold로 복구할 수 없다 → 고해상도 추론이 필요할 수 있다

### 개선 후보 4: Anomaly heatmap + SAM mask refinement

**아이디어**
SAM mask 선택과 boost 방식을 heatmap 정보로 개선한다.
- DetSeg-S: SAM 3개 후보 중 **heatmap과 가장 잘 맞는 mask** 선택, heatmap 평균이 낮은 mask 제거
- DetSeg-R: `+1.0` 상수 boost → `boost × SAM_mask × normalize(base_score)` soft fusion

**왜 필요한가**
- 현재 SAM 3개 후보 중 첫 번째를 고정 선택하고 SAM의 `iou_scores`도 버린다
- SAM은 box FP에도 mask를 만든다
- box boost는 box 내부 배경까지 올린다

**수정할 코드 위치**
- `SamPromptMixin.sam_predict_hf` `:299-318` (`outputs.iou_scores`도 반환하도록)
- `SamPromptMixin.sam_masks_from_boxes` `:320-335` (후보 선택 로직)
- `DetSegR._boost_anomaly_score` `:1363-1386` (soft fusion)
- `DetSegS.predict` `:1749-1769` (mask 필터)

**구현 방법**
1. `sam_predict_hf`에서 `pred_masks`(N×3×H×W)와 `iou_scores`(N×3)를 모두 반환
2. 선택 규칙 비교: (a) 첫 번째(현재) (b) SAM iou 최대 (c) heatmap 평균 최대 (d) box clip 후 (b)
3. mask 필터: `mean(heatmap[mask]) < τ` 또는 `area(mask)/area(box) > 1.2` → 제거
4. DetSeg-R: `anomaly + boost * mask * norm(anomaly)` 등 2~3개 변형

**필요한 추가 입력**
- SAM 가중치(`facebook/sam-vit-base`, 자동 다운로드)
- heatmap 결합을 하려면 **신뢰할 수 있는 score map이 필요하다** (외부 M2A map 또는 학습된 seg branch). 현재 internal heatmap은 사용 불가 (발견 #1)

**난이도**: 중
**졸업프로젝트 적합도**: 중상 (heatmap 확보가 선행 조건)
**예상 위험**
- score map 확보에 시간이 오래 걸릴 수 있다 → 일정이 빠듯하면 (a)(b)(d)처럼 heatmap 없는 변형만 진행

### 개선 후보 5: Query set 개선

**아이디어**
- (A) **FP 억제용 negative ID prompt**: `manhole, road marking, puddle, shadow, curb` 같은 도로 위 흔한 FP를 ID class로 추가
- (B) **known hazard 그룹**: `traffic cone, tire, cardboard box, animal, debris`를 별도 그룹으로 추가하고, 이 그룹은 OoD로 유지

**왜 필요한가**
- ID 집합이 Cityscapes 19 class에 고정되어 있어, 이 밖의 정상 물체/노면 패턴이 모두 OoD 후보가 된다
- 반대로 "자주 나오는 위험물"은 text로 직접 찾는 편이 objectness보다 신뢰도가 높을 수 있다

**수정할 코드 위치**
- prompt: `datasets.py` 각 dataset의 `METAINFO['classes']` (`:157` 등) / `src/run_detseg_single_image.py` `SetDetSegPrompt`
- `_ie_nms` `:522-524` (threshold 표 길이·의미)
- `DetSeg.predict` `:961`, `DetSegR.predict` `:1616` (hazard label 유지)

**구현 방법**
1. 개선 후보 2에서 threshold 표를 설정화 (선행 조건)
2. (A) 실험: 19 class vs 19+negative → FP 수 / FPR95 비교
3. (B) 실험: hazard 그룹 추가 → hazard label은 강화 제외 + 최종 유지 → recall 비교
4. **UN branch 출력 변화 측정**: 같은 이미지에서 prompt만 바꿨을 때 label-0 box 목록이 얼마나 바뀌는지 (발견 #3 검증)

**필요한 추가 입력**: 없음
**난이도**: 코드 하 / 실험 해석 중
**졸업프로젝트 적합도**: 중
**예상 위험**
- UN branch가 19-class prompt로만 학습되어 prompt 변경 시 분포가 달라질 수 있다
- 평가셋 GT 물체 이름을 prompt에 넣으면 사실상 "정답을 알려주는" 실험이 되어 open-world 주장이 약해진다 → 보고서에 반드시 명시하고 (A) 위주로 진행하는 것을 권장

---

## 7. 최종 답변

### 우리 졸업프로젝트에서 가장 현실적인 개선 주제는?

**"Road-aware IE-NMS & re-ranking: 도로 문맥과 물체 크기를 고려한 DetSeg OoD box 후처리 개선"**
(= 개선 후보 1 + 2 + 3을 하나의 후처리 모듈로 묶은 것)

### 이유

1. **재학습이 필요 없다.** 수정 범위가 `models.py`의 `_ie_nms`와 `predict` 후처리 몇십 줄이다
2. **논문의 핵심 모듈(IE-NMS)을 직접 다룬다.** "실행만 해봤다"가 아니라 "핵심 로직을 분석하고 약점을 고쳤다"는 서사가 성립한다
3. **코드에 근거가 있다.** road-overlap filter가 이미 있지만 꺼져 있고 약점(label 0 포함, 좌상단 모서리 기준, external 모드의 가짜 road)이 있다 → "기존 구현의 한계 분석 → 개선"으로 쓰기 좋다
4. **R/S 공통 효과.** box 단계 개선이라 DetSeg-S(평가가 쉬움)와 DetSeg-R(논문 지표) 모두에 적용된다
5. **자율주행 관점의 동기가 명확하다.** "주행 영역 위의 작은 물체를 놓치지 않는다"
6. **평가 도구가 이미 있다.** `eval_box_recall.py`(면적별 recall), `BinaryMaskAnomalyIoUMetric`(DetSeg-S), `AnomalyMetricLoad`(DetSeg-R)

### 바로 다음에 읽고 수정할 파일 (순서)

1. **데이터 준비 (선행 조건)**: FS Lost&Found(100장), RoadAnomaly(60장)를 `external/DetSeg-official/data/`에 배치. README의 Data Preparation 구조를 따른다
2. **읽기**: `mmdet/detseg_utils/models.py`
   - `:482-575` `predict_by_feat`, `_ie_nms`, `_predict_universal_by_feat_single`
   - `:843-965` `DetSeg.predict`
   - `:1388-1408`, `:1582-1624` road filter와 DetSeg-R 후처리
3. **베이스라인 측정**: `tools/eval_box_recall.py`(면적별 recall) + `configs/detseg/detseg-s_swin-b.py`(IoU/F1)
4. **첫 수정 (실험용 분기에서)**: `_ie_nms`의 하드코딩 값을 인자화 → 동작이 원본과 동일한지 확인 → 보호 규칙 R1~R3 추가
5. **다음 수정**: road mask helper(위치 prior → GDINO road box → mmseg 모델 순서) + soft re-ranking
6. **우리 스크립트 수정**: `src/run_detseg_single_image.py`를 external 또는 DetSeg-S config 기반으로 바꾸고, heatmap 출력은 "score map 있을 때만" 저장 (발견 #1)

> 권장: `external/DetSeg-official`은 gitignore 대상이므로, 수정본은 (a) DetSeg repo를 fork해 branch로 관리하거나, (b) 우리 repo의 `src/`에 `DetSegR`/`DetSegS` 하위 클래스(예: `DetSegSRoadAware`)를 새로 등록하고 config만 바꾸는 방식이 재현성에 좋다. **(b)를 추천**한다: 원본 코드를 건드리지 않고 ablation을 config로 전환할 수 있다.

---

## 8. 보고서용 요약 문장

> DetSeg는 Grounding DINO 기반 open-vocabulary detector에 class-agnostic universal query 300개와 objectness head를 추가한 구조이다. 입력 이미지는 Cityscapes 19개 class 이름 앞에 "objects" 토큰을 붙인 text prompt와 함께 처리되며, 900개의 ID query는 text 유사도로 알려진 class의 box를, 300개의 universal query는 objectness 점수로 class와 무관한 물체 후보를 예측한다. 두 query 집합은 decoder self-attention에서 서로 차단되어 독립적으로 동작한다.
>
> 두 branch의 출력은 IE-NMS(In-distribution Enhanced NMS)로 결합된다. IE-NMS는 class별 임계값(stuff 계열 0.2, thing 계열 0.5)을 넘는 ID box의 점수를 두 배로 올린 뒤 class-agnostic NMS(IoU 0.5)를 적용하여, ID 물체와 겹치는 universal 후보를 제거한다. 이후 label 0(universal/objects)이면서 점수가 0.2를 넘는 box만 OoD 후보로 남긴다.
>
> DetSeg-R은 외부 방법(Mask2Anomaly, RbA 등)이 생성한 pixel 단위 anomaly score map에서 OoD box 내부 pixel에 상수 1.0을 더해, box 밖의 false positive를 상대적으로 낮춘다. DetSeg-S는 OoD box를 SAM의 box prompt로 사용해 threshold 탐색 없이 이진 anomaly mask를 생성한다.
>
> 코드 분석 결과 다음 한계를 확인하였다. (1) IE-NMS의 임계값과 IoU가 19개 class에 맞춰 하드코딩되어 있고, pole·traffic sign 등 물체형 class도 낮은 임계값(0.2)을 사용하여 이와 유사한 OoD 물체가 억제될 수 있다. (2) 최종 box threshold가 물체 크기와 무관하여 작은 도로 낙하물이 탈락하기 쉽다. (3) 도로 영역을 고려하는 필터가 구현되어 있으나 기본적으로 비활성화되어 있고, 도로 영역 정의에 OoD 후보 자신이 포함되는 등 실효성이 제한적이다. (4) 논문에서 제안한 VPHM은 배포 코드에서 동작 가능한 형태로 제공되지 않는다.
>
> 본 프로젝트는 재학습 없이 적용할 수 있는 후처리 개선, 즉 도로 문맥과 물체 크기를 고려한 IE-NMS 보호 규칙 및 universal box re-ranking을 제안하고, box recall(면적 구간별), DetSeg-S IoU/F1, DetSeg-R AUROC/AP/FPR95로 그 효과를 검증한다.

---

## 9. 확인 필요 목록

| 항목 | 확인 방법 |
|---|---|
| `query_text_mask`가 실제로 무시되는지 | decoder layer에 hook을 걸어 UN query의 text attention weight 확인, 또는 prompt 변경 시 label-0 box 변화 측정 |
| label 0 box 중 ID branch('objects' token) 비율 | `predict_by_feat`에서 `results_id.labels==0` 개수를 출력 |
| internal config 로드 시 seg_decoder missing key 경고 | `load_checkpoint` 로그 확인 |
| `road_label_ids=arange(3)`에 label 0을 넣은 의도 | 논문 ablation 설명 대조, 저자 issue |
| 외부 score map(M2A 등) 입수 경로 | 저자 repo issue / 각 방법 공식 repo에서 생성 |
| 원본 해상도(2048) 추론 시 4070 Ti 메모리 | 실제 실행 |
| 논문의 "ID query/UN query" 용어와 코드 변수명 대응 | 논문 3장 원문 대조 |
| obj365 / mix_data config의 역할 | 논문 ablation 표 대조 |
