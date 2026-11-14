# DetSeg-S baseline on Road Anomaly (EPFL official WebP release).
# Differences from the official detseg-s_swin-b.py:
#   - dataset: RoadAnomalyDataset (frame_list.json + frames/*.webp)
#   - UnifyGT: Road Anomaly labels anomaly as 2, but BinaryMaskAnomalyIoUMetric
#     treats only gt == 1 as OoD. Label values checked: {0, 2}, no void.
# Run from external/DetSeg-official so data/ and ckpts/ paths resolve.
_base_ = '../external/DetSeg-official/configs/detseg/detseg-s_swin-b.py'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='UnifyGT', label_map={0: 0, 2: 1}),
    dict(type='ConcatPrompt'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'custom_entities', 'tokens_positive')),
]

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _delete_=True,
        type='RoadAnomalyDataset',
        data_root='data/RoadAnomaly',
        pipeline=test_pipeline))
val_dataloader = test_dataloader
