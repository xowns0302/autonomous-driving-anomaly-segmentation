# DetSeg-S on SMIYC_Anomaly_val (Mask2Anomaly-distributed validation set: images/, labels_masks/, GT {0,1,255}).
# Only the dataset differs from the official detseg-s_swin-b.py.
# Run from external/DetSeg-official so data/ and ckpts/ paths resolve.
_base_ = '../external/DetSeg-official/configs/detseg/detseg-s_swin-b.py'

test_dataloader = dict(
    dataset=dict(
        data_root='data/SMIYC_Anomaly_val',
        img_suffix='.png',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')))
val_dataloader = test_dataloader
