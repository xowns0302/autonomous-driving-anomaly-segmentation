# E4. DetSeg-S ablation (논문 Table 6, 부분)

- 날짜: 2026-10-16
- 단계: Phase 1 (재현)

## 1. 목적

DetSeg-S의 구성 요소(UN query, IE-NMS)가 각각 얼마나 기여하는지, 논문 Table 6과 같은 경향인지 확인한다.

## 2. 설정

| 행 | 논문 Table 6 | 우리 구현 | 가능 여부 |
|---|---|---|---|
| ① baseline | Grounding DINO + prompt "objects" | **MM-Grounding-DINO Swin-B 원본 가중치**(DetSeg 학습 시작점, 공식 README 링크)에 텍스트 `('objects',)`만 입력, score > 0.2 box → SAM (`src/gdino_objects_baseline.py`) | ✅ |
| ② UN query, mask 없음 | 별도 학습 모델 | – | ❌ 가중치가 하나뿐이라 재현 불가 |
| ③ UN query + mask, IE-NMS 없음 | | 공개 가중치, IE-NMS의 점수 강화를 끔 (bar 1.0 → 일반 class-agnostic NMS) (`src/eval_detseg_s_diagnostics.py --stuff-bar 1.0 --thing-bar 1.0`) | ✅ |
| ④ 전체 | | 공개 가중치 그대로 (= E1) | ✅ |

공통: τ = 0.2, SAM ViT-B, IoU = 이미지별 IoU 평균 (E1과 같은 정의). ④는 진단 스크립트로 다시 돌려 E1 값과 정확히 같음을 확인했다.

## 3. 실행 명령

```bash
cd external/DetSeg-official && export PYTHONPATH=.
# ①
python ../../src/gdino_objects_baseline.py --config ../../configs/detseg-s_$ds.py \
  --checkpoint ckpts/grounding_dino_swin-b_pretrain_all-f9818a7c.pth --out-dir work_dirs/E4/${ds}_gdino_objects
# ③, ④
python ../../src/eval_detseg_s_diagnostics.py --config ../../configs/detseg-s_$ds.py --checkpoint $CK \
  --out-dir work_dirs/E4/${ds}_no_ienms --no-save-masks --stuff-bar 1.0 --thing-bar 1.0
python ../../src/eval_detseg_s_diagnostics.py --config ../../configs/detseg-s_$ds.py --checkpoint $CK \
  --out-dir work_dirs/E4/${ds}_full --no-save-masks
```

## 4. 결과 (IoU, 괄호는 논문)

| 행 | UN query | Attn mask | IE-NMS | Road Anomaly | FS L&F | FS Static | SMIYC Anomaly | SMIYC Obstacle |
|---|---|---|---|---|---|---|---|---|
| ① | – | – | – | 53.82 (61.21) | 2.72 (19.39) | 14.44 (38.46) | **85.62** | 39.15 |
| ③ | ✓ | (✓) | – | 69.44 (60.89) | 4.99 (7.29) | 14.20 (36.60) | 75.06 | 77.93 |
| ④ | ✓ | (✓) | ✓ | **76.46** (68.35) | **14.26** (32.36) | **33.51** (83.34) | 69.12 | 77.93 |

(✓): 공개 코드에서 query-text mask는 실제로 적용되지 않는다 (01 문서 3.1절). self-attention mask만 적용

이미지당 평균 OoD box 수:

| 행 | RA | FS L&F | FS Static | SMIYC A | SMIYC O |
|---|---|---|---|---|---|
| ① objects | 2.82 | 3.53 | 3.07 | 1.50 | 3.57 |
| ③ IE-NMS 없음 | 11.55 | 15.80 | 23.97 | 13.80 | 3.00 |
| ④ 전체 | 10.20 | 10.51 | 17.00 | 12.80 | 2.67 |

## 5. 해석

1. **논문과 같은 경향:** RA·FS L&F·FS Static에서 ① < ③ < ④. UN query와 IE-NMS가 순서대로 성능을 올린다. 특히 **IE-NMS의 효과가 크다** (FS L&F 4.99 → 14.26, FS Static 14.20 → 33.51). 논문 Table 6의 "IE-NMS가 Fishyscapes에서 결정적"이라는 결론과 같다
2. **하지만 IE-NMS 후에도 box가 너무 많다.** FS L&F는 IE-NMS가 box를 15.8 → 10.5개로 줄이지만, 정답 물체는 이미지당 2개 정도다. IE-NMS가 **일부만 작동**하는 상태다. 논문 수준(32.36)에 도달하지 못하는 이유로 보인다 → A2·A3에서 bar 값과 연결해 확인
3. **SMIYC Anomaly는 반대 경향:** "objects" baseline(85.62)이 가장 좋고, UN query(75.06)와 IE-NMS(69.12)를 더할수록 나빠진다. 큰 물체 1~2개가 있는 장면이라 baseline은 box 1.5개로 깔끔하지만, DetSeg는 12.8개를 남긴다. E3에서 DetSeg-R이 이 데이터셋에서 악화된 것과 같은 원인이다 (box 오탐)
4. **SMIYC Obstacle에서는 IE-NMS 효과가 0이다** (③ = ④ = 77.93). 시골길·악천후 장면에 ID 물체(차량 등)가 거의 없어서 지울 것이 없다. 반대로 UN query의 효과는 매우 크다 (39.15 → 77.93): 작은 장애물은 "objects" 텍스트로는 잘 안 잡히고 UN query가 잡는다 (논문 Figure 5의 주장과 일치)
5. **① baseline이 논문보다 낮다** (FS L&F 2.72 vs 19.39). 논문의 baseline 구현(텍스트 구성, threshold, Grounding DINO 가중치 버전)이 명시되어 있지 않아 차이 원인은 확정할 수 없다

## 6. 결과 파일

- `results/E4/{dataset}_{gdino_objects,no_ienms,full}/summary.json`, `per_image.json`
