"""
Created on Wed Mar 26 2025 by LKH
Auto generate annotations for mask detection(Video)
"""
import logging
import os
from pathlib import Path
import cv2

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import eval_metrics, show_mask, show_box, annotation_generator


def iou(box1, boxes2):
    """
        Calculate the intersection over union between two boxes
        : param boxes1: xyxy
        : param boxes2: xyxy
    """
    xy_max = np.minimum(boxes2[:, 2:], box1[0, 2:])
    xy_min = np.maximum(boxes2[:, :2], box1[0, :2])
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0] * inter[:, 1]

    area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    return inter / (area_box1 + area_boxes2 - inter)


class GeneratorVideo(object):
    def __init__(self, yolo_model, sam_predictor, data_loader, conf, device):
        self.yolo_model = yolo_model
        self.sam_predictor = sam_predictor
        self.data_loader = data_loader
        # video dir(determine if it's the same video)
        self.video_dir = None
        # inference_state
        self.inference_state = None
        # last frame categories/bboxes(object tracker)
        self.last_categories = []
        self.last_bboxes = []
        self.num_classes = self.data_loader.dataset.num_classes
        self.classes = self.data_loader.dataset.classes
        self.palette = self.data_loader.dataset.palette
        self.conf = conf
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self, ann_dir: str):
        self.logger.info("YOLO-SAM2 Video Auto_Mask Generator")
        self._reset_metrics()
        tbar = tqdm(self.data_loader, ncols=130)
        for batch_idx, (data, data_dir, target, categories_id, bboxes, images_id, images_index) in enumerate(tbar):

            if self.yolo_model is not None:
                result = self.yolo_model.predict(source=data, conf=self.conf, device=self.device)
                data = []
                for r in result:
                    data.append(cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB))
                    categories_id.append(r.boxes.cls.cpu().numpy().astype(np.int32))
                    bboxes.append(r.boxes.xyxy.cpu().numpy())

            indices = [i for i, category_id in enumerate(categories_id) if len(category_id) > 0]
            data = [data[i] for i in indices]
            data_dir = [data_dir[i] for i in indices]
            target = [target[i] for i in indices]
            categories_id = [categories_id[i] for i in indices]
            bboxes = [bboxes[i] for i in indices]
            images_id = [images_id[i] for i in indices]
            images_index = [images_index[i] for i in indices]
            if len(categories_id) == 0:
                continue

            # determine if it's the same video(initial inference_state)
            if self.video_dir != data_dir[0]:
                self.video_dir = data_dir[0]
                self.inference_state = self.sam_predictor.init_state(video_path=self.video_dir)

            # prompt of the batch frame
            for i in range(len(bboxes)):
                for j, bbox in enumerate(bboxes[i]):
                    _, object_ids, masks = self.sam_predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=images_index[i],
                        obj_id=j + 1,
                        box=bbox,
                    )

            # run propagation throughout the video and collect the results in a dict
            # video_segments = {}  # video_segments contains the per-frame segmentation results
            idx_batch = []
            masks_batch = []
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_predictor.propagate_in_video(
                    self.inference_state):
                if out_frame_idx in images_index:
                    idx_batch.append(out_obj_ids)
                    masks_batch.append((out_mask_logits > 0.0).cpu().numpy())

            # post-precess
            # get the current frame detection boxes and categories using IOU matching of the front and back frames
            new_bboxes = []
            new_categories_id = []
            new_masks_batch = []
            for category_id, bbox, masks, image_index in zip(categories_id, bboxes, masks_batch, images_index):
                new_bbox = []
                masks_index = []
                # get the new boxes from object masks
                for i in range(len(masks)):
                    mask = np.squeeze(masks[i])
                    coords = np.argwhere(mask)
                    if coords.size == 0:
                        continue
                    masks_index.append(i)
                    # get the row and col index
                    cols = coords[:, 0]
                    rows = coords[:, 1]

                    # calculate bounding box
                    x1, x2 = rows.min(), rows.max()
                    y1, y2 = cols.min(), cols.max()
                    new_bbox.append([x1, y1, x2, y2])
                new_bbox = np.array(new_bbox, dtype=np.float32)
                new_bboxes.append(new_bbox)
                # remove the invalid masks
                new_masks_batch.append(masks[masks_index])
                # get object the categories using IOU matching of the front and back frames
                if image_index == 0:
                    new_category_id = []
                    for i, box in enumerate(new_bbox):
                        box_iou = iou(box[np.newaxis, :], bbox)
                        max_index = np.argmax(box_iou)
                        new_category_id.append(category_id[max_index])
                    new_category_id = np.array(new_category_id, dtype=np.int32)
                    new_categories_id.append(new_category_id)
                    self.last_bboxes = new_bbox
                    self.last_categories = new_category_id
                else:
                    if len(bbox) == 0:
                        bbox = self.last_bboxes
                        category_id = self.last_categories
                    elif len(self.last_bboxes) != 0:
                        bbox = np.vstack((bbox, self.last_bboxes))
                        category_id = np.hstack((category_id, self.last_categories))
                    new_category_id = []
                    for i, box in enumerate(new_bbox):
                        box_iou = iou(box[np.newaxis, :], bbox)
                        max_index = np.argmax(box_iou)
                        new_category_id.append(category_id[max_index])
                    new_category_id = np.array(new_category_id, dtype=np.int32)
                    new_categories_id.append(new_category_id)
                    self.last_bboxes = new_bbox
                    self.last_categories = new_category_id

            # Auto_Mask
            for masks, category_id, image_id, image in zip(new_masks_batch, new_categories_id, images_id, data):
                annotation_generator(masks, category_id, image_id, image.shape[-2], image.shape[-3], ann_dir, self.classes)

            tbar.set_description('Auto_Mask')


