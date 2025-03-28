"""
    Created on Fri Mar 21 2025 by LKH
    Cascade YOLOv8 and SAM2 to predict the corresponding objects
    based on the YOLOv8 detection boxes to get the labelled object masks.
"""
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import numpy as np
import torch

from argparse import ArgumentParser
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dataloaders import VID
from core import Generator

if __name__ == '__main__':
    parser = ArgumentParser(description='YOLO-SAM2 Auto Annotation Generator')
    parser.add_argument(
        "-c",
        "--config",
        default="config_imagenet_vid.json",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
        type=int,
        help="batch size"
    )
    parser.add_argument(
        "-conf",
        "--conf_thres",
        default=0.25,
        type=float,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        # default='yolo/yolov8/predictions.json',
        type=str,
        help="path to prompt directory"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    np.random.seed(2024)

    config = json.load(open(args.config))

    if args.prompt is None:
        yolo_model = YOLO(config["arch"]["yolo_weights"])
    else:
        yolo_model = None

    sam2_model = build_sam2(config["arch"]["sam_config"], config["arch"]["sam_weights"], device=device)

    sam_predictor = SAM2ImagePredictor(sam2_model)

    data_loader = VID(config, args.prompt, args.conf_thres)

    mask_generator = Generator(yolo_model, sam_predictor, data_loader, args.conf_thres, device)

    mask_generator.generate(config["auto_mask"]["save_dir"])




