import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DetSeg single-image inference and save visualizations."
    )

    parser.add_argument(
        "--detseg-root",
        default="external/DetSeg-official",
        help="Path to DetSeg official repository.",
    )
    parser.add_argument(
        "--config",
        default="configs/detseg/detseg-r_swin-b_internal.py",
        help="Config path relative to DetSeg root.",
    )
    parser.add_argument(
        "--checkpoint",
        default="ckpts/detseg_swin-b_coco_20260623-207453f5.pth",
        help="Checkpoint path relative to DetSeg root.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Input image path. Absolute path or relative path from project root.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/qualitative_results/script_outputs",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.20,
        help="Score threshold for visualizing predicted boxes.",
    )

    return parser.parse_args()


def resolve_path(base: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base / path


def setup_detseg_import(detseg_root: Path):
    """
    DetSeg official repo의 mmdet package를 import할 수 있게 경로를 추가한다.
    """
    detseg_root_str = str(detseg_root.resolve())
    if detseg_root_str not in sys.path:
        sys.path.insert(0, detseg_root_str)


def register_single_image_prompt_transform():
    """
    DetSeg 단일 이미지 inference에서 필요한 text prompt를 pipeline에 넣기 위한 transform 등록.
    DetSeg는 Grounding DINO 계열이라 image만 넣으면 text key가 없어서 KeyError가 발생한다.
    """
    from mmcv.transforms.base import BaseTransform
    from mmdet.registry import TRANSFORMS

    if "SetDetSegPrompt" in TRANSFORMS.module_dict:
        return

    @TRANSFORMS.register_module()
    class SetDetSegPrompt(BaseTransform):
        """Add Cityscapes-style text prompt for single-image DetSeg inference."""

        def transform(self, results: dict) -> dict:
            results["text"] = (
                "road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "traffic light",
                "traffic sign",
                "vegetation",
                "terrain",
                "sky",
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            )
            results["custom_entities"] = True
            return results


def insert_prompt_transform(cfg):
    """
    config의 test pipeline에서 ConcatPrompt 바로 앞에 SetDetSegPrompt를 삽입한다.
    """
    pipeline = cfg.test_dataloader.dataset.pipeline

    already_inserted = any(step.get("type") == "SetDetSegPrompt" for step in pipeline)
    if already_inserted:
        return cfg

    insert_idx = None
    for i, step in enumerate(pipeline):
        if step.get("type") == "ConcatPrompt":
            insert_idx = i
            break

    if insert_idx is None:
        raise RuntimeError("ConcatPrompt not found in test pipeline.")

    pipeline.insert(insert_idx, dict(type="SetDetSegPrompt"))
    return cfg


def save_box_visualization(image_bgr, result, output_path: Path, score_thr: float):
    """
    result.pred_instances의 bbox를 원본 이미지 위에 그려 저장한다.
    """
    h, w = image_bgr.shape[:2]
    boxed = image_bgr.copy()

    instances = result.pred_instances
    bboxes = instances.bboxes.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy()

    kept = 0

    for box, score in zip(bboxes, scores):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            boxed,
            f"obj {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        kept += 1

    cv2.imwrite(str(output_path), boxed)
    return len(bboxes), kept


def make_anomaly_heatmap(image_bgr, result):
    """
    result.anomaly_scores를 heatmap 이미지로 변환한다.
    """
    h, w = image_bgr.shape[:2]

    score_tensor = result.anomaly_scores.data

    if isinstance(score_tensor, torch.Tensor):
        score_map = score_tensor.detach().cpu().float().numpy()
    else:
        score_map = np.asarray(score_tensor)

    score_map = np.squeeze(score_map)

    score_map_resized = cv2.resize(
        score_map,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    min_v = float(score_map_resized.min())
    max_v = float(score_map_resized.max())

    if max_v - min_v < 1e-8:
        norm = np.zeros_like(score_map_resized, dtype=np.uint8)
    else:
        norm = ((score_map_resized - min_v) / (max_v - min_v) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return heatmap, min_v, max_v


def main():
    args = parse_args()

    project_root = Path.cwd()
    detseg_root = resolve_path(project_root, args.detseg_root)
    config_path = resolve_path(detseg_root, args.config)
    checkpoint_path = resolve_path(detseg_root, args.checkpoint)
    image_path = resolve_path(project_root, args.image)
    output_dir = resolve_path(project_root, args.output_dir)

    if not detseg_root.exists():
        raise FileNotFoundError(f"DetSeg root not found: {detseg_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    setup_detseg_import(detseg_root)

    from mmengine.config import Config
    from mmdet.apis import init_detector, inference_detector

    register_single_image_prompt_transform()

    cfg = Config.fromfile(str(config_path))
    cfg = insert_prompt_transform(cfg)

    print("DetSeg root:", detseg_root)
    print("Config:", config_path)
    print("Checkpoint:", checkpoint_path)
    print("Image:", image_path)
    print("Output dir:", output_dir)
    print("Score threshold:", args.score_thr)

    print("loading model...")
    model = init_detector(cfg, str(checkpoint_path), device="cuda:0")

    print("running inference...")
    result = inference_detector(model, str(image_path))

    print("loading original image...")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    stem = image_path.stem
    thr_tag = f"thr{int(args.score_thr * 100):03d}"

    box_path = output_dir / f"{stem}_boxes_{thr_tag}.png"
    heatmap_path = output_dir / f"{stem}_anomaly_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    total_boxes, kept_boxes = save_box_visualization(
        image_bgr=image_bgr,
        result=result,
        output_path=box_path,
        score_thr=args.score_thr,
    )

    heatmap, min_v, max_v = make_anomaly_heatmap(image_bgr, result)
    cv2.imwrite(str(heatmap_path), heatmap)

    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(str(overlay_path), overlay)

    print("done")
    print("saved files:")
    print("-", box_path)
    print("-", heatmap_path)
    print("-", overlay_path)
    print("num boxes total:", total_boxes)
    print("num boxes kept:", kept_boxes)
    print("anomaly score min:", min_v)
    print("anomaly score max:", max_v)


if __name__ == "__main__":
    main()