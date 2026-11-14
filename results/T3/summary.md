# Latency summary: results/T3

```
NVIDIA GeForce RTX 4070 Ti, 616.92, 12282 MiB
Model name:                              AMD Ryzen 5 3600 6-Core Processor
Sun Oct 18 02:47:42 KST 2026
65aca96
```

30 images after 5 warm-up, 3 repeats, batch 1. Values: mean ± std over repeats (ms/image).

| Dataset | 방법 | DetSeg 모델 | 데이터 읽기 | M2A 점수 지도 | **이미지당 합계** | 검출 | SAM | C5 | C7 (DINOv2) |
|---|---|---|---|---|---|---|---|---|---|
| fs_lostfound | M2A 단독 | – | – | 498 ± 6 | **498 ± 6** | – | – | – | – |
| fs_lostfound | DetSeg-S (공개본) | 419 ± 3 | 114 ± 4 | – | **532 ± 4** | 163 | 227 | – | – |
| fs_lostfound | DetSeg-S (논문 방식 bar) | 353 ± 1 | 112 ± 1 | – | **464 ± 1** | 162 | 163 | – | – |
| fs_lostfound | DetSeg-S + C5 | 430 ± 6 | 112 ± 1 | 498 | **1040 ± 6** | 167 | 194 | 40.2 | – |
| fs_lostfound | DetSeg-S + C7 | 433 ± 1 | 113 ± 0 | – | **546 ± 1** | 161 | 109 | – | 135.1 |
| fs_lostfound | DetSeg-S + C8 (C5+C7) | 495 ± 2 | 113 ± 0 | 498 | **1106 ± 2** | 165 | 130 | 36.8 | 134.7 |
| fs_lostfound | M2A + DetSeg-R (공개본) | 200 ± 1 | 79 ± 0 | 498 | **777 ± 1** | 162 | – | – | – |
| fs_lostfound | M2A + DetSeg-R + C5 | 240 ± 1 | 79 ± 0 | 498 | **817 ± 1** | 166 | – | 35.8 | – |
| fs_lostfound | M2A + DetSeg-R + C8 | 376 ± 2 | 79 ± 0 | 498 | **954 ± 3** | 165 | – | 40.7 | 134.4 |
| road_anomaly | M2A 단독 | – | – | 405 ± 1 | **405 ± 1** | – | – | – | – |
| road_anomaly | DetSeg-S (공개본) | 386 ± 1 | 47 ± 0 | – | **433 ± 1** | 176 | 186 | – | – |
| road_anomaly | DetSeg-S (논문 방식 bar) | 383 ± 0 | 47 ± 0 | – | **430 ± 1** | 175 | 183 | – | – |
| road_anomaly | DetSeg-S + C5 | 399 ± 2 | 47 ± 0 | 405 | **852 ± 2** | 180 | 184 | 10.0 | – |
| road_anomaly | DetSeg-S + C7 | 433 ± 0 | 47 ± 0 | – | **480 ± 0** | 173 | 163 | – | 72.8 |
| road_anomaly | DetSeg-S + C8 (C5+C7) | 449 ± 0 | 47 ± 0 | 405 | **901 ± 0** | 178 | 164 | 9.3 | 72.6 |
| road_anomaly | M2A + DetSeg-R (공개본) | 206 ± 1 | 34 ± 0 | 405 | **646 ± 1** | 176 | – | – | – |
| road_anomaly | M2A + DetSeg-R + C5 | 221 ± 1 | 34 ± 0 | 405 | **660 ± 1** | 180 | – | 9.7 | – |
| road_anomaly | M2A + DetSeg-R + C8 | 291 ± 0 | 34 ± 0 | 405 | **731 ± 0** | 178 | – | 9.5 | 73.1 |
| smiyc_obstacle | M2A 단독 | – | – | 483 ± 2 | **483 ± 2** | – | – | – | – |
| smiyc_obstacle | DetSeg-S (공개본) | 429 ± 1 | 116 ± 0 | – | **545 ± 1** | 176 | 229 | – | – |
| smiyc_obstacle | DetSeg-S (논문 방식 bar) | 429 ± 2 | 116 ± 0 | – | **545 ± 1** | 175 | 229 | – | – |
| smiyc_obstacle | DetSeg-S + C5 | 469 ± 1 | 116 ± 1 | 483 | **1068 ± 1** | 179 | 228 | 37.7 | – |
| smiyc_obstacle | DetSeg-S + C7 | 488 ± 2 | 116 ± 0 | – | **604 ± 2** | 173 | 164 | – | 127.5 |
| smiyc_obstacle | DetSeg-S + C8 (C5+C7) | 535 ± 2 | 116 ± 0 | 483 | **1134 ± 2** | 178 | 165 | 40.6 | 127.9 |
| smiyc_obstacle | M2A + DetSeg-R (공개본) | 208 ± 0 | 84 ± 0 | 483 | **775 ± 1** | 176 | – | – | – |
| smiyc_obstacle | M2A + DetSeg-R + C5 | 252 ± 1 | 84 ± 0 | 483 | **819 ± 1** | 180 | – | 39.1 | – |
| smiyc_obstacle | M2A + DetSeg-R + C8 | 378 ± 4 | 84 ± 0 | 483 | **945 ± 4** | 178 | – | 39.7 | 128.1 |
| smiyc_anomaly | M2A 단독 | – | – | 394 ± 2 | **394 ± 2** | – | – | – | – |
| smiyc_anomaly | DetSeg-S (공개본) | 380 ± 5 | 34 ± 1 | – | **414 ± 6** | 179 | 175 | – | – |
| smiyc_anomaly | DetSeg-S (논문 방식 bar) | 352 ± 1 | 34 ± 0 | – | **387 ± 1** | 173 | 155 | – | – |
| smiyc_anomaly | DetSeg-S + C5 | 368 ± 1 | 34 ± 0 | 394 | **796 ± 1** | 178 | 156 | 9.2 | – |
| smiyc_anomaly | DetSeg-S + C7 | 408 ± 0 | 34 ± 0 | – | **442 ± 0** | 173 | 150 | – | 60.0 |
| smiyc_anomaly | DetSeg-S + C8 (C5+C7) | 424 ± 2 | 34 ± 0 | 394 | **852 ± 2** | 178 | 151 | 9.5 | 60.6 |
| smiyc_anomaly | M2A + DetSeg-R (공개본) | 204 ± 1 | 23 ± 0 | 394 | **620 ± 1** | 174 | – | – | – |
| smiyc_anomaly | M2A + DetSeg-R + C5 | 219 ± 1 | 23 ± 0 | 394 | **635 ± 1** | 179 | – | 9.5 | – |
| smiyc_anomaly | M2A + DetSeg-R + C8 | 279 ± 1 | 23 ± 0 | 394 | **695 ± 1** | 178 | – | 10.2 | 60.2 |

## 단계별 (DetSeg-S 공개본)

| 단계 | fs_lostfound | road_anomaly | smiyc_obstacle | smiyc_anomaly |
|---|---|---|---|---|
| data | 113.5 | 46.6 | 116.3 | 34.4 |
| backbone | 64.1 | 69.2 | 69.1 | 69.1 |
| neck | 1.1 | 1.1 | 1.1 | 1.1 |
| text | 0.3 | 0.3 | 0.3 | 0.3 |
| encoder | 56.7 | 64.1 | 64.1 | 65.9 |
| query_sel | 7.0 | 7.2 | 7.2 | 7.6 |
| decoder | 23.6 | 23.5 | 23.5 | 24.5 |
| head_excl_ie_nms | 7.1 | 7.0 | 7.0 | 7.4 |
| ie_nms | 3.6 | 3.5 | 3.6 | 3.7 |
| sam_total | 226.7 | 186.2 | 229.2 | 175.2 |
| sam_prep | 0.0 | 0.0 | 0.0 | 0.0 |
| sam_image_encoder | 0.0 | 0.0 | 0.0 | 0.0 |
| sam_mask_decoder | 0.0 | 0.0 | 0.0 | 0.0 |
| sam_post | 0.0 | 0.0 | 0.0 | 0.0 |
| other_in_model | 26.6 | 22.6 | 21.7 | 23.1 |
| model_total | 418.8 | 386.5 | 428.9 | 379.7 |
| total | 532.3 | 433.0 | 545.2 | 414.1 |
