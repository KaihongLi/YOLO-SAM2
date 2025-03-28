"""
Created on Wed Mar 26 2025 by LKH
Predict object masks and save visualisation results
"""
import logging
import os
from pathlib import Path
import cv2

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import eval_metrics, show_mask, show_box


class Predictor(object):
    def __init__(self, yolo_model, sam_predictor, data_loader, conf, device):
        self.yolo_model = yolo_model
        self.sam_predictor = sam_predictor
        self.data_loader = data_loader
        self.num_classes = self.data_loader.dataset.num_classes
        self.classes = self.data_loader.dataset.classes
        self.palette = self.data_loader.dataset.palette
        self.conf = conf
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

    def inference(self, visual_dir: str):
        self.logger.info("YOLO+SAM2 Segmentation inference")
        tbar = tqdm(self.data_loader, ncols=130)
        for batch_idx, (data, _, categories_id, bboxes, images_id) in enumerate(tbar):

            if self.yolo_model is not None:
                result = self.yolo_model.predict(source=data, conf=self.conf, device=self.device)
                data = []
                for r in result:
                    data.append(cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB))
                    categories_id.append(r.boxes.cls.cpu().numpy().astype(np.int64))
                    bboxes.append(r.boxes.xyxy.cpu().numpy())

            indices = [i for i, category_id in enumerate(categories_id) if len(category_id) > 0]
            data = [data[i] for i in indices]
            categories_id = [categories_id[i] for i in indices]
            bboxes = [bboxes[i] for i in indices]
            images_id = [images_id[i] for i in indices]
            if len(categories_id) == 0:
                continue
            self.sam_predictor.set_image_batch(data)

            masks_batch, scores_batch, _ = self.sam_predictor.predict_batch(
                None,
                None,
                box_batch=bboxes,
                multimask_output=False
            )

            # visual
            for image, boxes, masks, category_id, scores, image_id in zip(data, bboxes, masks_batch, categories_id, scores_batch, images_id):
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                for mask in masks:
                    show_mask(mask.squeeze(0) if mask.ndim == 3 else mask, plt.gca(), random_color=True)
                for i in range(len(boxes)):
                    show_box(boxes[i], plt.gca(), category_id[i], scores[i], self.classes, self.palette)
                plt.axis('off')
                masks_path = os.path.join(visual_dir, image_id + '.png')
                if not os.path.exists(Path(masks_path).parent):
                    os.makedirs(Path(masks_path).parent)
                plt.savefig(masks_path, dpi=300, bbox_inches='tight')
                plt.close()

            tbar.set_description('Visual')

