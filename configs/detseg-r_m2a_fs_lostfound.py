# E3: Mask2Anomaly score maps + DetSeg-R on FS_LostFound. E2 (M2A alone): --cfg-options model.box_score_thr=1.1
# Score maps: results/score_maps/m2a/FS_LostFound/{image name}.npy (src/m2a_save_scores.py)
# Run from external/DetSeg-official.
_base_ = './_detseg-r_m2a_base.py'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=False, with_seg=True),

    dict(type='ConcatPrompt'),
    dict(type='GetAnomalyScoreMap', data_path='../../results/score_maps/m2a/FS_LostFound'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'anomaly_score_map',
                    'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities')),
]
test_dataloader = dict(batch_size=1, num_workers=2, dataset=dict(_delete_=True, **dict(type='FSLostAndFoundDataset', data_root='data/FS_LostFound', img_suffix='.png', pipeline=test_pipeline, data_prefix=dict(img_path='images', seg_map_path='labels_masks'))))
val_dataloader = test_dataloader
