# 03. 환경과 데이터 (Phase 0)

- 작성: 2026-10-16
- 관련: [02_scope_and_plan.md](02_scope_and_plan.md) Phase 0

---

## 1. 하드웨어·OS

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti (12GB) |
| OS | WSL2 Ubuntu 22.04 (kernel 6.18 microsoft-standard-WSL2) |
| 디스크 여유 | 약 938GB |
| 시스템 CUDA 컴파일러(nvcc) | 없음 → 필요한 환경에만 conda로 설치 |

## 2. DetSeg 환경 (conda `detseg`)

| 항목 | 값 |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.1.0+cu118 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| mmdet (repo 내장) / mmsegmentation | 3.3.0 / 1.2.2 |
| transformers | 4.46.3 |
| SAM | `facebook/sam-vit-base` (HuggingFace 캐시) |
| 공식 코드 | `external/DetSeg-official`, HEAD `a1aca17b` (2026-09-29) |
| checkpoint | `ckpts/detseg_swin-b_coco_20260623-207453f5.pth`, SHA-256 앞자리 `207453f5ce82cf03` (파일명 해시와 일치) |

논문(4.1절·README)의 테스트 환경: Python 3.8, PyTorch 2.1.0 + CUDA 11.8, MMCV 2.1.0, MMEngine 0.10.7 → **일치**

## 3. Mask2Anomaly 환경 (conda `m2a`)

| 항목 | 값 |
|---|---|
| 코드 | `external/Mask2Anomaly` (공식 repo, HEAD `5c84283`, 2024-04-28) |
| 가중치 | `best_contrastive.pth` (공식 Google Drive, `~/datasets/m2a_validation_dl/Mask2Anomaly/`) |
| Python / PyTorch | 3.8 / 2.1.0+cu118 |
| CUDA 컴파일러 | conda `nvidia/label/cuda-11.8.0` cuda-nvcc (env 내부) |
| detectron2, MSDeformAttn op | 소스 빌드 (진행 중) |

M2A 점수 계산 (공식 `anomaly_utils/anomaly_inference.py`):
- anomaly score = `1 − max softmax(19 class)`, 원본과 좌우 반전 결과의 평균
- "anomaly mask query" 출력으로 일부 픽셀을 0으로 만드는 보정이 들어 있다
- **OoD 픽셀이 없는 이미지는 평가에서 제외한다** (`if 1 not in np.unique(ood_gts): continue`) → FS Static 30장 중 10장이 빠진다. 해석 시 주의

## 4. 데이터

원본은 `~/datasets/`, DetSeg에서는 `external/DetSeg-official/data/`의 symlink로 연결.

| Dataset | 장수 | 해상도 | GT 값 | OoD 없는 이미지 | 출처 | `data/` 이름 | config |
|---|---|---|---|---|---|---|---|
| Road Anomaly | 60 | 1280×720 | 0, **2** | 0 | EPFL 공식 `RoadAnomaly.zip` (WebP) | `RoadAnomaly` | `configs/detseg-s_road_anomaly.py` (UnifyGT 2→1) |
| FS L&F | 100 | 2048×1024 | 0, 1, 255 | 1 | M2A 배포 `Validation_Dataset.zip` | `FS_LostFound` | `configs/detseg-s_fs_lostfound.py` |
| FS Static | 30 | 2048×1024 | 0, 1, 255 | **10** | 같음 (`fs_static`) | `FS_Static` | `configs/detseg-s_fs_static.py` |
| SMIYC Anomaly val | 10 | 1280×720 | 0, 1, 255 | 0 | 같음 (`RoadAnomaly21`) | `SMIYC_Anomaly_val` | `configs/detseg-s_smiyc_anomaly.py` |
| SMIYC Obstacle val | 30 | 1920×1080 | 0, 1, 255 | 0 | 같음 (`RoadObsticle21`) | `SMIYC_Obstacle_val` | `configs/detseg-s_smiyc_obstacle.py` |
| Cityscapes val | 500 | 2048×1024 | – | – | 공식 (사용자 계정) | 다운로드 중 | IE-NMS bar 보정용 |

- 모든 config로 dataset을 빌드해 장수와 GT 값을 확인했다 (Road Anomaly는 UnifyGT 후 {0, 1})
- Mask2Anomaly 배포본은 Mask2Anomaly 논문의 평가 데이터와 같은 파일이다. DetSeg 논문이 어떤 버전을 썼는지는 명시되지 않았다 (FS Static은 "일관된 버전"이라고만 언급)

다운로드:
- Road Anomaly: `wget https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly.zip`
- M2A 배포본: https://drive.google.com/drive/folders/1eQhmPbKSZrN1AsieY9KFchfll7XC1_SF → `Validation_Dataset.zip`, `best_contrastive.pth`
- Cityscapes: https://www.cityscapes-dataset.com → `gtFine_trainvaltest.zip`, `leftImg8bit_trainvaltest.zip`
