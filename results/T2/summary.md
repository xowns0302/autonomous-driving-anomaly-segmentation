# Latency summary: results/T2

```
NVIDIA GeForce RTX 4070 Ti, 616.92, 12282 MiB
Model name:                              AMD Ryzen 5 3600 6-Core Processor
Sat Oct 17 23:24:34 KST 2026
6c10670
```

30 images after 5 warm-up, 3 repeats, batch 1. Values: mean ± std over repeats (ms/image).

| Dataset | 방법 | DetSeg 모델 | 데이터 읽기 | M2A 점수 지도 | **이미지당 합계** | 검출 | SAM | C5 | C7 (DINOv2) |
|---|---|---|---|---|---|---|---|---|---|
| fs_lostfound | M2A 단독 | – | – | 498 ± 6 | **498 ± 6** | – | – | – | – |
| fs_lostfound | DetSeg-S (공개본) | 696 ± 7 | 112 ± 1 | – | **808 ± 8** | 191 | 437 | – | – |
| fs_lostfound | DetSeg-S (논문 방식 bar) | 490 ± 5 | 113 ± 1 | – | **603 ± 6** | 189 | 258 | – | – |
| fs_lostfound | DetSeg-S + C5 | 579 ± 10 | 113 ± 1 | 498 | **1189 ± 11** | 202 | 299 | 32.5 | – |
| fs_lostfound | DetSeg-S + C7 | 620 ± 2 | 114 ± 3 | – | **734 ± 2** | 188 | 200 | – | 197.8 |
| fs_lostfound | DetSeg-S + C8 (C5+C7) | 725 ± 8 | 113 ± 0 | 498 | **1336 ± 8** | 202 | 248 | 36.0 | 201.7 |
| fs_lostfound | M2A + DetSeg-R (공개본) | 229 ± 0 | 79 ± 0 | 498 | **806 ± 0** | 191 | – | – | – |
| fs_lostfound | M2A + DetSeg-R + C5 | 279 ± 0 | 80 ± 0 | 498 | **857 ± 1** | 203 | – | 37.6 | – |
| fs_lostfound | M2A + DetSeg-R + C8 | 471 ± 4 | 79 ± 0 | 498 | **1048 ± 4** | 201 | – | 32.1 | 201.0 |
| road_anomaly | M2A 단독 | – | – | 405 ± 1 | **405 ± 1** | – | – | – | – |
| road_anomaly | DetSeg-S (공개본) | 630 ± 7 | 49 ± 0 | – | **679 ± 8** | 208 | 374 | – | – |
| road_anomaly | DetSeg-S (논문 방식 bar) | 567 ± 3 | 48 ± 0 | – | **615 ± 3** | 203 | 323 | – | – |
| road_anomaly | DetSeg-S + C5 | 614 ± 8 | 49 ± 0 | 405 | **1068 ± 9** | 215 | 343 | 11.8 | – |
| road_anomaly | DetSeg-S + C7 | 677 ± 2 | 48 ± 0 | – | **725 ± 2** | 203 | 309 | – | 126.1 |
| road_anomaly | DetSeg-S + C8 (C5+C7) | 724 ± 3 | 48 ± 0 | 405 | **1178 ± 3** | 213 | 329 | 9.3 | 130.0 |
| road_anomaly | M2A + DetSeg-R (공개본) | 233 ± 0 | 34 ± 0 | 405 | **672 ± 0** | 203 | – | – | – |
| road_anomaly | M2A + DetSeg-R + C5 | 252 ± 0 | 34 ± 0 | 405 | **691 ± 0** | 212 | – | 10.2 | – |
| road_anomaly | M2A + DetSeg-R + C8 | 381 ± 1 | 34 ± 0 | 405 | **821 ± 1** | 213 | – | 10.1 | 127.8 |
| smiyc_obstacle | M2A 단독 | – | – | 483 ± 2 | **483 ± 2** | – | – | – | – |
| smiyc_obstacle | DetSeg-S (공개본) | 538 ± 2 | 117 ± 1 | – | **655 ± 3** | 203 | 304 | – | – |
| smiyc_obstacle | DetSeg-S (논문 방식 bar) | 542 ± 2 | 117 ± 0 | – | **659 ± 2** | 202 | 308 | – | – |
| smiyc_obstacle | DetSeg-S + C5 | 593 ± 11 | 116 ± 0 | 483 | **1192 ± 11** | 211 | 311 | 37.5 | – |
| smiyc_obstacle | DetSeg-S + C7 | 703 ± 2 | 117 ± 1 | – | **819 ± 2** | 202 | 293 | – | 178.2 |
| smiyc_obstacle | DetSeg-S + C8 (C5+C7) | 771 ± 12 | 119 ± 1 | 483 | **1373 ± 13** | 216 | 304 | 37.0 | 181.5 |
| smiyc_obstacle | M2A + DetSeg-R (공개본) | 236 ± 1 | 84 ± 0 | 483 | **803 ± 1** | 203 | – | – | – |
| smiyc_obstacle | M2A + DetSeg-R + C5 | 286 ± 1 | 84 ± 0 | 483 | **853 ± 1** | 213 | – | 39.9 | – |
| smiyc_obstacle | M2A + DetSeg-R + C8 | 458 ± 1 | 84 ± 0 | 483 | **1024 ± 1** | 212 | – | 37.2 | 177.2 |
| smiyc_anomaly | M2A 단독 | – | – | 394 ± 2 | **394 ± 2** | – | – | – | – |
| smiyc_anomaly | DetSeg-S (공개본) | 618 ± 1 | 36 ± 0 | – | **654 ± 1** | 206 | 367 | – | – |
| smiyc_anomaly | DetSeg-S (논문 방식 bar) | 572 ± 4 | 36 ± 0 | – | **608 ± 4** | 205 | 323 | – | – |
| smiyc_anomaly | DetSeg-S + C5 | 613 ± 3 | 36 ± 0 | 394 | **1043 ± 3** | 215 | 341 | 10.9 | – |
| smiyc_anomaly | DetSeg-S + C7 | 667 ± 5 | 36 ± 0 | – | **703 ± 5** | 206 | 302 | – | 114.9 |
| smiyc_anomaly | DetSeg-S + C8 (C5+C7) | 719 ± 2 | 36 ± 0 | 394 | **1148 ± 2** | 215 | 334 | 9.9 | 115.9 |
| smiyc_anomaly | M2A + DetSeg-R (공개본) | 232 ± 2 | 23 ± 0 | 394 | **649 ± 2** | 203 | – | – | – |
| smiyc_anomaly | M2A + DetSeg-R + C5 | 251 ± 0 | 23 ± 0 | 394 | **667 ± 0** | 212 | – | 9.5 | – |
| smiyc_anomaly | M2A + DetSeg-R + C8 | 368 ± 1 | 23 ± 0 | 394 | **785 ± 1** | 214 | – | 9.5 | 114.6 |

## 단계별 (DetSeg-S 공개본)

| 단계 | fs_lostfound | road_anomaly | smiyc_obstacle | smiyc_anomaly |
|---|---|---|---|---|
| data | 112.1 | 48.5 | 116.5 | 35.6 |
| backbone | 63.9 | 69.2 | 69.1 | 69.1 |
| neck | 1.1 | 1.2 | 1.0 | 1.1 |
| text | 28.5 | 29.0 | 27.6 | 28.4 |
| encoder | 56.7 | 64.4 | 63.8 | 64.2 |
| query_sel | 7.1 | 7.8 | 7.1 | 7.5 |
| decoder | 23.4 | 25.1 | 23.4 | 24.6 |
| head_excl_ie_nms | 7.0 | 7.6 | 7.0 | 7.3 |
| ie_nms | 3.7 | 3.9 | 3.6 | 3.8 |
| sam_prep | 130.0 | 80.7 | 137.5 | 67.7 |
| sam_image_encoder | 119.9 | 119.9 | 119.2 | 119.9 |
| sam_mask_decoder | 12.5 | 18.3 | 8.2 | 17.9 |
| sam_post | 175.1 | 154.8 | 38.7 | 161.1 |
| other_in_model | 65.3 | 46.2 | 30.0 | 43.9 |
| model_total | 696.3 | 630.0 | 538.2 | 618.3 |
| total | 808.4 | 678.5 | 654.8 | 654.0 |
