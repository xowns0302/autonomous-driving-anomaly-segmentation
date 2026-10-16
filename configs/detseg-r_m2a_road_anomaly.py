# E3: Mask2Anomaly score maps + DetSeg-R on RoadAnomaly. E2 (M2A alone): --cfg-options model.box_score_thr=1.1
# Score maps: results/score_maps/m2a/RoadAnomaly/{image name}.npy (src/m2a_save_scores.py)
# Run from external/DetSeg-official.
_base_ = './_detseg-r_m2a_base.py'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=False, with_seg=True),
    dict(type='UnifyGT', label_map={0: 0, 2: 1}),
    dict(type='ConcatPrompt'),
    dict(type='GetAnomalyScoreMap', data_path='../../results/score_maps/m2a/RoadAnomaly'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'anomaly_score_map',
                    'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities')),
]
test_dataloader = dict(batch_size=1, num_workers=2, dataset=dict(_delete_=True, **dict(type='RoadAnomalyDataset', data_root='data/RoadAnomaly', pipeline=test_pipeline)))
val_dataloader = test_dataloader
