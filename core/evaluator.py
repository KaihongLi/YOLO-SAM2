"""
Created on Wed Mar 26 2025 by LKH
Evaluate the effectiveness of YOLO and SAM2 cascades
"""
import logging
import os
from pathlib import Path
import cv2

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import eval_metrics, show_mask, show_box, annotation_generator


class Evaluator(object):
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

    def evaluate(self, visual_dir: str, ann_dir: str):
        self.logger.info("YOLO+SAM2 Segmentation Evaluating")
        self._reset_metrics()
        tbar = tqdm(self.data_loader, ncols=130)
        for batch_idx, (data, target, categories_id, bboxes, images_id) in enumerate(tbar):

            if self.yolo_model is not None:
                result = self.yolo_model.predict(source=data, conf=self.conf, device=self.device)
                data = []
                for r in result:
                    data.append(cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB))
                    categories_id.append(r.boxes.cls.cpu().numpy().astype(np.int32))
                    bboxes.append(r.boxes.xyxy.cpu().numpy())

            indices = [i for i, category_id in enumerate(categories_id) if len(category_id) > 0]
            data = [data[i] for i in indices]
            target = [target[i] for i in indices]
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

            # Visual
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

            # mIOU, cls-IOU
            outputs = []
            for category_id, bbox, masks, scores in zip(categories_id, bboxes, masks_batch, scores_batch):
                indices = np.argsort(np.squeeze(scores))
                category_id = category_id[indices]
                masks = masks[indices]
                output = np.zeros_like(masks[0])
                for i in range(len(masks)):
                    output[masks[i] == 1] = category_id[i] + 1
                outputs.append(np.squeeze(output))
            height = outputs[0].shape[0]
            width = outputs[0].shape[1]
            for i, output in enumerate(outputs):
                if output.shape[0] != height or output.shape[1] != width:
                    print(images_id[i])
            seg_metrics = eval_metrics(outputs, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)

            # Print Info
            pixAcc, mIoU, _ = self._get_seg_metrics().values()
            tbar.set_description('EVAL | PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(pixAcc, mIoU))

            # Auto_Mask
            for masks, category_id, image_id, image in zip(masks_batch, categories_id, images_id, data):
                annotation_generator(masks, category_id, image_id, image.shape[-2], image.shape[-3], ann_dir, self.classes)

        seg_metrics = self._get_seg_metrics()

        log = {
            **seg_metrics
        }

        # LOGGING INFO
        for k, v in log.items():
            if isinstance(v, dict):
                infos = ''
                for key, value in v.items():
                    infos += f'{key}:{value} '
                self.logger.info(f'         {str(k):15s}: {infos}')
            else:
                self.logger.info(f'         {str(k):15s}: {v}')

        return log

    def _reset_metrics(self):
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }
