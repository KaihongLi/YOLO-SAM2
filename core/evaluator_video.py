"""
Created on Wed Mar 26 2025 by LKH
Evaluate the effectiveness of YOLO and SAM2 Video cascades
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


class EvaluatorVideo(object):
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

    def evaluate(self, visual_dir: str, ann_dir: str):
        self.logger.info("YOLO-SAM2 Video Segmentation Evaluating")
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
                    if bbox.shape[0] == 0:
                        bbox = self.last_bboxes
                        category_id = self.last_categories
                    elif self.last_bboxes.shape[0] != 0:
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

            # Visual
            # for image, boxes, masks, category_id, scores, image_id in zip(data, new_bboxes, new_masks_batch, new_categories_id, scores_batch, images_id):
            #     plt.figure(figsize=(8, 6))
            #     plt.imshow(image)
            #     for mask in masks:
            #         show_mask(mask.squeeze(0) if mask.ndim == 3 else mask, plt.gca(), random_color=True)
            #     for i in range(len(boxes)):
            #         show_box(boxes[i], plt.gca(), category_id[i], scores[i], self.classes, self.palette)
            #     plt.axis('off')
            #     masks_path = os.path.join(visual_dir, image_id + '.png')
            #     if not os.path.exists(Path(masks_path).parent):
            #         os.makedirs(Path(masks_path).parent)
            #     plt.savefig(masks_path, dpi=300, bbox_inches='tight')
            #     plt.close()

            # calculate mIOU, cls-IOU
            outputs = []
            index = 0
            for category_id, bbox, masks in zip(new_categories_id, new_bboxes, new_masks_batch):
                output = np.zeros_like(target[index], dtype=np.int32)
                masks = np.squeeze(masks, axis=1)
                for i in range(len(masks)):
                    if masks[i].shape[0] != target[index].shape[0] or masks[i].shape[1] != target[index].shape[1]:
                        new_mask = np.array(masks[i], dtype=np.uint8)
                        new_mask = cv2.resize(new_mask, (int(target[index].shape[1]), int(target[index].shape[0])),
                                              interpolation=cv2.INTER_CUBIC)
                        new_mask = np.array(new_mask, dtype=bool)
                        output[new_mask == 1] = category_id[i] + 1
                    else:
                        output[masks[i] == 1] = category_id[i] + 1
                outputs.append(np.squeeze(output))
                index += 1
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
            # for masks, category_id, image_id, image in zip(new_masks_batch, new_categories_id, images_id, data):
            #     annotation_generator(masks, category_id, image_id, image.shape[-2], image.shape[-3], ann_dir, self.classes)

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
