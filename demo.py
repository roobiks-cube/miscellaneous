# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob
import json

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def convert_structure(obj):
    if isinstance(obj, dict):
        return {k: convert_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_structure(v) for v in obj]
    return to_numpy(obj)


def print_structure(obj, prefix=""):
    if isinstance(obj, dict):
        print(f"{prefix}dict with keys: {list(obj.keys())}")
        for k, v in obj.items():
            if hasattr(v, "shape"):
                print(f"{prefix}{k}: shape={v.shape}")
            else:
                print(f"{prefix}{k}: {type(v)}")
    elif isinstance(obj, list):
        print(f"{prefix}list length={len(obj)}")
        for i, item in enumerate(obj[:3]):  # print first 3 items only
            print(f"{prefix}[{i}] -> {type(item)}")
            if isinstance(item, dict):
                for k, v in item.items():
                    if hasattr(v, "shape"):
                        print(f"{prefix}  {k}: shape={v.shape}")
                    else:
                        print(f"{prefix}  {k}: {type(v)}")
    else:
        print(f"{prefix}{type(obj)}")


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None

    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )

    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )

    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    print(f"Found {len(images_list)} image(s).")

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        print("\n===== MODEL OUTPUT =====")
        print("Image:", image_path)

        if outputs is None:
            print("No outputs returned.")
            continue

        outputs_np = convert_structure(outputs)

        print("Top-level type:", type(outputs_np))
        print_structure(outputs_np)

        # More detailed preview
        if isinstance(outputs_np, dict):
            if "pred_joints" in outputs_np:
                print("\n🧍 pred_joints (first 5):")
                print(np.round(outputs_np["pred_joints"][0][:5], 3))

            if "pred_vertices" in outputs_np:
                print("\n🔺 pred_vertices (first 5):")
                print(np.round(outputs_np["pred_vertices"][0][:5], 3))

            if "pred_cam" in outputs_np:
                print("\n📷 pred_cam:")
                print(np.round(outputs_np["pred_cam"], 3))

        elif isinstance(outputs_np, list) and len(outputs_np) > 0:
            first = outputs_np[0]
            if isinstance(first, dict):
                if "pred_joints" in first:
                    print("\n🧍 first item pred_joints (first 5):")
                    print(np.round(first["pred_joints"][0][:5], 3))

                if "pred_vertices" in first:
                    print("\n🔺 first item pred_vertices (first 5):")
                    print(np.round(first["pred_vertices"][0][:5], 3))

                if "pred_cam" in first:
                    print("\n📷 first item pred_cam:")
                    print(np.round(first["pred_cam"], 3))

        # Save JSON
        json_ready = convert_structure(outputs)

        def make_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_serializable(v) for v in obj]
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        json_path = f"{output_folder}/{os.path.basename(image_path)[:-4]}.json"
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(json_ready), f)

        # Render image
        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        cv2.imwrite(
            f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img.astype(np.uint8),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

Environment Variables:
SAM3D_MHR_PATH: Path to MHR asset
SAM3D_DETECTOR_PATH: Path to human detection model folder
SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
SAM3D_FOV_PATH: Path to fov estimation model folder
""",
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_MHR_PATH)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    args = parser.parse_args()

    main(args)