"""PROWL-style prototype bank (Sinhamahapatra et al., "Finding Dino", WACV).

Builds, for each of the 19 Cityscapes classes, L prototype vectors: the mean
DINOv2 ViT-S/14 patch feature inside the ground-truth mask of one object
(paper Eq. 1). Objects come from Cityscapes *train* (no overlap with any
evaluation set):
  * thing classes (person .. bicycle): gtFine instance masks
  * stuff classes (road .. sky): the class region of one image (no instances)
Objects are sampled with a fixed seed, at most one per image per class, with
a minimum area so that each covers several 14x14 patches.

Step 1 (CPU): choose objects and list the images to extract
  python src/prowl_bank.py select --gt-root ~/datasets/cityscapes --out results/P1/bank_objects.json
Step 2: extract those images from leftImg8bit_trainvaltest.zip (see docs)
Step 3 (GPU): python src/prowl_bank.py build --gt-root ~/datasets/cityscapes \
      --objects results/P1/bank_objects.json --out results/P1/prototypes.npz
"""
import argparse
import glob
import json
import os
import random

import numpy as np

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'train', 'motorcycle', 'bicycle')
LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
THING_IDS = set(LABEL_IDS[11:])
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # DINOv2 (ImageNet) normalisation
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def grid_size(H, W, short=518, patch=14):
    """Resize so the short side is `short`, both sides multiples of the patch size."""
    s = short / min(H, W)
    return int(round(H * s / patch)) * patch, int(round(W * s / patch)) * patch


def patch_features(model, img_rgb, device='cuda', short=518):
    """L2-normalised patch tokens, shape (h, w, D), and the resized (H', W')."""
    import torch
    from PIL import Image
    H, W = img_rgb.shape[:2]
    Hr, Wr = grid_size(H, W, short)
    x = np.asarray(Image.fromarray(img_rgb).resize((Wr, Hr), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    with torch.no_grad():
        tok = model(pixel_values=t).last_hidden_state[0, 1:]           # drop CLS
    tok = torch.nn.functional.normalize(tok, dim=-1)
    return tok.reshape(Hr // 14, Wr // 14, -1), (Hr, Wr)


def load_model(device='cuda'):
    from transformers import AutoModel
    return AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()


def select(args):
    from PIL import Image
    rng = random.Random(args.seed)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.gt_root), 'gtFine/train/*/*_instanceIds.png')))
    rng.shuffle(files)
    chosen = {c: [] for c in CLASSES}
    for f in files:
        if all(len(v) >= args.per_class for v in chosen.values()):
            break
        inst = np.array(Image.open(f))
        name = os.path.basename(f).replace('_gtFine_instanceIds.png', '')
        for ci, (c, lid) in enumerate(zip(CLASSES, LABEL_IDS)):
            if len(chosen[c]) >= args.per_class:
                continue
            if lid in THING_IDS:
                ids = [v for v in np.unique(inst) if v // 1000 == lid and v >= 1000]
                ids = [v for v in ids if (inst == v).sum() >= args.min_area]
                if ids:
                    chosen[c].append(dict(image=name, kind='instance', value=int(rng.choice(ids))))
            else:
                if (inst == lid).sum() >= args.min_area:    # stuff pixels keep their labelId
                    chosen[c].append(dict(image=name, kind='semantic', value=lid))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(per_class=args.per_class, min_area=args.min_area, seed=args.seed, objects=chosen),
              open(args.out, 'w'), indent=1)
    images = sorted({o['image'] for v in chosen.values() for o in v})
    for c in CLASSES:
        print(f'{c:14s} {len(chosen[c])}')
    print('images needed:', len(images))
    with open(args.out.replace('.json', '_images.txt'), 'w') as fh:
        for im in images:
            fh.write(f'leftImg8bit/train/{im.split("_")[0]}/{im}_leftImg8bit.png\n')


def build(args):
    from PIL import Image
    root = os.path.expanduser(args.gt_root)
    sel = json.load(open(args.objects))
    model = load_model()
    protos = np.zeros((len(CLASSES), sel['per_class'], 384), np.float32)
    for ci, c in enumerate(CLASSES):
        for li, o in enumerate(sel['objects'][c]):
            city = o['image'].split('_')[0]
            img = np.array(Image.open(f'{root}/leftImg8bit/train/{city}/{o["image"]}_leftImg8bit.png').convert('RGB'))
            inst = np.array(Image.open(f'{root}/gtFine/train/{city}/{o["image"]}_gtFine_instanceIds.png'))
            mask = inst == o['value']
            feat, (Hr, Wr) = patch_features(model, img)
            h, w = feat.shape[:2]
            # fraction of each patch covered by the mask
            m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.BOX)) / 255.0
            wgt = (m >= 0.5).astype(np.float32)
            if wgt.sum() == 0:
                wgt = (m == m.max()).astype(np.float32)
            v = (feat.cpu().numpy() * wgt[..., None]).sum((0, 1)) / wgt.sum()
            protos[ci, li] = v / (np.linalg.norm(v) + 1e-8)
        print(f'{c:14s} done', flush=True)
    np.savez_compressed(args.out, prototypes=protos, classes=np.array(CLASSES))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('select')
    a.add_argument('--gt-root', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--per-class', type=int, default=20)
    a.add_argument('--min-area', type=int, default=4000)
    a.add_argument('--seed', type=int, default=0)
    b = sub.add_parser('build')
    b.add_argument('--gt-root', required=True)
    b.add_argument('--objects', required=True)
    b.add_argument('--out', required=True)
    args = p.parse_args()
    select(args) if args.cmd == 'select' else build(args)


if __name__ == '__main__':
    main()
