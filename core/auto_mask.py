"""
Created on Wed Mar 26 2025 by LKH
Auto generate annotations for mask detection
"""
import logging
import cv2
import numpy as np
from tqdm import tqdm

from utils import eval_metrics, annotation_generator


class Generator(object):
    def __init__(self, yolo_model, sam_predictor, data_loader, conf, device):
        self.yolo_model = yolo_model
        self.sam_predictor = sam_predictor
        self.data_loader = data_loader
        self.num_classes = self.data_loader.dataset.num_classes
        self.classes = self.data_loader.dataset.classes
        self.conf = conf
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self,  ann_dir: str):
        self.logger.info("YOLO+SAM2 Segmentation Auto_Mask Generator")
        tbar = tqdm(self.data_loader, ncols=130)
        for batch_idx, (data, _, categories_id, bboxes, images_id) in enumerate(tbar):

            if self.yolo_model is not None:
                result = self.yolo_model.predict(source=data, conf=self.conf, device=self.device)
                data = []
                for r in result:
                    data.append(cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB))
                    categories_id.append(r.boxes.cls.cpu().numpy().astype(np.int32))
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

            # Auto_Mask
            for masks, category_id, image_id, image in zip(masks_batch, categories_id, images_id, data):
                annotation_generator(masks, category_id, image_id, image.shape[-2], image.shape[-3], ann_dir, self.classes)

