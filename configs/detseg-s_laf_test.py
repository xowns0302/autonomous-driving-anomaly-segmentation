# DetSeg-S on the Lost and Found test split prepared by src/prepare_laf.py (plan docs/05 §10).
# images/ (symlinks), labels_masks/ GT {0 road, 1 anomaly, 255 ignore}. Only the dataset differs.
_base_ = '../external/DetSeg-official/configs/detseg/detseg-s_swin-b.py'

test_dataloader = dict(
    dataset=dict(
        data_root='data/LAF_test',
        img_suffix='.png',
        data_prefix=dict(img_path='images', seg_map_path='labels_masks')))
val_dataloader = test_dataloader
