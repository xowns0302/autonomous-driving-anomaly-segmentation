# Shared settings for E2 (M2A alone) and E3 (M2A + DetSeg-R).
# Dataset-specific configs set `ds_root`, `ds_img_suffix`, `ds_img_dir`, `ds_unify_gt`
# and call nothing else; everything below is rebuilt from those variables.
_base_ = '../external/DetSeg-official/configs/detseg/detseg-r_swin-b_external.py'

model = dict(box_score_thr=0.2)   # official DetSeg-R value (E3). E2 overrides to 1.1 (no box kept).

val_evaluator = [
    dict(type='AnomalyMetricLoad', prefix='pixel'),   # AUROC / AP / FPR95 (Table 1)
    dict(type='OracleThresholdAnomalyIoUMetric', prefix='oracle'),  # best IoU over thresholds, step 0.01 (Table 2)
]
test_evaluator = val_evaluator
