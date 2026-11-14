"""Speed-ups that must not change any result (T3). Applied as monkeypatches; the
official DetSeg files are untouched.

O1 SAM post-processing: the official code (SamPromptMixin.sam_predict_hf +
   sam_masks_from_boxes, models.py:299-335) upsamples all 3 multimask outputs
   of every box to the original size on the CPU and then keeps mask 0. Here
   only mask 0 is upsampled, on the GPU, with the same two bilinear steps and
   the same threshold (transformers SamImageProcessor._post_process_masks_pt).
O2 SAM pre-processing: the official code runs the processor twice; the second
   call re-processes the whole image only to transform the boxes. Here the
   image is processed once and the boxes are transformed with the processor's
   own _normalize_coordinates.
O3 Text encoder: the prompt is identical for every image, so the BERT output
   is computed once per prompt and a fresh copy is returned each call
   (DetSeg modifies text_dict in place, models.py:1026).
   Image cache: the RGB image read for C7 (DINOv2) is reused by SAM.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_IMAGE_CACHE = {}


def load_rgb(path):
    """PIL RGB image, decoded once per path (only the latest image is kept)."""
    if path not in _IMAGE_CACHE:
        _IMAGE_CACHE.clear()
        _IMAGE_CACHE[path] = Image.open(path).convert('RGB')
    return _IMAGE_CACHE[path]


def patch_sam():
    from mmdet.detseg_utils.models import SamPromptMixin

    def sam_masks_from_boxes(self, img_path, boxes, output_shape, device):
        if len(boxes) == 0:
            return torch.zeros((1, *output_shape), device=device)
        sam_model = self._sam_on_device(device)
        processor = self.sam_processor
        raw = load_rgb(img_path)
        inputs = processor(raw, return_tensors='pt').to(device)
        emb = sam_model.get_image_embeddings(inputs['pixel_values'])
        orig = inputs['original_sizes'][0].tolist()
        reshaped = inputs['reshaped_input_sizes'][0].tolist()
        b = processor._normalize_coordinates(processor.target_size, boxes.cpu().numpy(), orig, is_bounding_box=True)
        input_boxes = torch.from_numpy(np.array([b])).to(device)
        with torch.no_grad():
            out = sam_model(image_embeddings=emb, input_boxes=input_boxes)
        m = out.pred_masks[0][:, :1].float()                                # (N, 1, 256, 256), mask 0 only
        pad = processor.image_processor.pad_size
        m = F.interpolate(m, (pad['height'], pad['width']), mode='bilinear', align_corners=False)
        m = m[..., :reshaped[0], :reshaped[1]]
        m = F.interpolate(m, tuple(orig), mode='bilinear', align_corners=False)
        masks = (m > 0.0)[:, 0].float()
        if masks.shape[-2:] != tuple(output_shape):
            masks = F.interpolate(masks.unsqueeze(1), size=output_shape, mode='bilinear',
                                  align_corners=False).squeeze(1)
        return masks
    SamPromptMixin.sam_masks_from_boxes = sam_masks_from_boxes


def patch_text_cache():
    from mmdet.models.language_models.bert import BertModel
    orig = BertModel.forward
    cache = {}

    def forward(self, captions, *a, **k):
        key = (id(self), tuple(captions))
        if key not in cache:
            cache[key] = orig(self, captions, *a, **k)
        return {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in cache[key].items()}
    BertModel.forward = forward


def patch_amp():
    """L1 (paper limitation 'Efficiency'): run the Grounding DINO detector in FP16
    (torch.autocast). Unlike O1-O3 this CAN change results; it is measured, not assumed.
    DetSeg-S: only the detection part (DetSeg.predict); SAM stays FP32.
    DetSeg-R: the whole predict (detection + box boost)."""
    from mmdet.detseg_utils.models import DetSeg, DetSegR

    def wrap(fn):
        def f(self, *a, **k):
            with torch.autocast('cuda', dtype=torch.float16):
                out = fn(self, *a, **k)
            for ds in out:                       # hand FP32 boxes/scores to the rules downstream
                pi = getattr(ds, 'pred_instances', None)
                if pi is not None:
                    for key in ('bboxes', 'scores'):
                        if key in pi and torch.is_floating_point(pi[key]):
                            pi[key] = pi[key].float()
            return out
        return f
    DetSeg.predict = wrap(DetSeg.predict)
    DetSegR.predict = wrap(DetSegR.predict)


def apply_all():
    patch_sam()
    patch_text_cache()


def box_proto_signal(feat, boxes, H, W, P):
    """Vectorised proto_box for many boxes (same definition as run_with_bars.proto_box).

    feat: (h, w, D) normalised DINOv2 patch features; boxes: (N, 4) in original pixels.
    Returns 1 - max cosine(mean feature in box, any prototype)."""
    h, w, D = feat.shape
    B = np.asarray(boxes, dtype=np.float64)
    x1 = np.clip(B[:, 0], 0, W); x2 = np.clip(B[:, 2], 0, W)
    y1 = np.clip(B[:, 1], 0, H); y2 = np.clip(B[:, 3], 0, H)
    gx1 = np.minimum((x1 / W * w).astype(int), w - 1); gx2 = np.ceil(x2 / W * w).astype(int)
    gy1 = np.minimum((y1 / H * h).astype(int), h - 1); gy2 = np.ceil(y2 / H * h).astype(int)
    gx2 = np.maximum(gx2, gx1 + 1); gy2 = np.maximum(gy2, gy1 + 1)
    ii = F.pad(feat.double().cumsum(0).cumsum(1), (0, 0, 1, 0, 1, 0))         # (h+1, w+1, D) integral
    gx1, gx2, gy1, gy2 = (torch.as_tensor(a, device=feat.device) for a in (gx1, gx2, gy1, gy2))
    s = ii[gy2, gx2] - ii[gy1, gx2] - ii[gy2, gx1] + ii[gy1, gx1]
    v = F.normalize((s / ((gy2 - gy1) * (gx2 - gx1))[:, None]).float(), dim=1)
    return (1 - (v @ P.reshape(-1, D).T).amax(1)).cpu().numpy()
