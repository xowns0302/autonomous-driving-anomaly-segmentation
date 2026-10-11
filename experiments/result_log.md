# Result Log

| Date | Exp | Method | Dataset | AUROC | AP | FPR95 | Notes |
|---|---|---|---|---|---|---|---|

## DetSeg 단일 이미지 Inference 실험 — Phase 8~9

### 날짜

2026-10-12

### 목적

DetSeg official code가 로컬 WSL2 + RTX 4070 Ti 환경에서 정상적으로 실행되는지 확인하고, 단일 도로 이미지에 대해 bbox 및 anomaly score map을 생성할 수 있는지 확인한다.

### 실행 환경

- OS: WSL2 Ubuntu 22.04
- Conda environment: detseg
- Python: 3.8.20
- GPU: NVIDIA GeForce RTX 4070 Ti
- PyTorch: 2.1.0+cu118
- MMCV: 2.1.0
- MMEngine: 0.10.7
- mmdet: 3.3.0

### 사용 코드

- Official repo: external/DetSeg-official
- Config: configs/detseg/detseg-r_swin-b_internal.py
- Checkpoint: ckpts/detseg_swin-b_coco_20260623-207453f5.pth

### 입력 이미지

- demo_inputs/road.png

### 단일 이미지 inference 수정 사항

기본 `inference_detector` 실행 시 `KeyError: 'text'`가 발생했다.

원인:

- DetSeg는 Grounding DINO 계열 구조를 사용하므로 text prompt가 필요하다.
- 단일 이미지 inference에서는 `text` key가 자동으로 생성되지 않는다.
- `ConcatPrompt` transform이 `results['text']`를 요구하면서 에러가 발생했다.

해결:

- `ConcatPrompt` 앞에 단일 이미지용 `SetDetSegPrompt` transform을 임시 삽입했다.
- Cityscapes 19개 class prompt를 입력했다.

### 결과

생성된 결과 파일:

- assets/qualitative_results/road_boxes.png
- assets/qualitative_results/road_anomaly_heatmap.png
- assets/qualitative_results/road_overlay.png

정량 로그:

- num boxes total: 13
- num boxes kept: 13
- anomaly score min: -18.806081771850586
- anomaly score max: -13.129426956176758

### 결과 해석

이번 실험은 full benchmark evaluation이 아니다.

즉 다음을 의미하지 않는다.

- Fishyscapes / RoadAnomaly 정량 평가 완료
- AUROC / AP / FPR95 재현 완료
- 논문 성능 재현 완료

이번 실험의 의미는 다음과 같다.

- DetSeg official code가 로컬 GPU 환경에서 실행 가능함을 확인했다.
- 단일 이미지 입력에 대해 bbox prediction을 생성할 수 있음을 확인했다.
- anomaly score map을 생성하고 heatmap/overlay 이미지로 저장할 수 있음을 확인했다.

### 다음 작업

- 결과 이미지 확인
- threshold 조정 실험
- inference script를 `src/` 또는 `experiments/` 아래에 정리
- dataset 기반 evaluation 가능 여부 검토

