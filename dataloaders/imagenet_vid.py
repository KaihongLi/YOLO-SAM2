"""
    Created on Wed Mar 19 2025 by LKH
    Thyroid Dataset
"""

from base import BaseDataSet, BaseDataLoader
import numpy as np
import os

from PIL import Image
import json

from .labels import _IMAGENET_VID_CLASSES
from utils import get_voc_palette


class VIDDataset(BaseDataSet):

    def __init__(self, num_classes, conf, **kwargs):
        self.num_classes = num_classes
        self.conf = conf
        self.palette = get_voc_palette(num_classes)
        self.classes = _IMAGENET_VID_CLASSES
        super(VIDDataset, self).__init__(**kwargs)

    def _set_files(self):
        """
            Load image filenames
        """
        self.root = os.path.join(self.root, 'test')  # voc dataset name
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')
        # get the prompt file
        if self.prompt_file is not None:
            with open(self.prompt_file) as f:
                predictions = json.load(f)
                for prediction in predictions:
                    if prediction["image_id"] not in self.prompts:
                        self.prompts[prediction["image_id"]] = [prediction]
                    else:
                        self.prompts[prediction["image_id"]].append(prediction)
            video_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + "_videos.txt")
            videos = [line.rstrip() for line in tuple(open(video_list, "r"))]
            for video in videos:
                video_dir = os.path.join(self.label_dir, video)
                image_list = [os.path.join(video, image.split('.')[0]) for image in os.listdir(video_dir)]
                image_list = [image_list[i:i+self.batch_size] for i in range(0, len(image_list), self.batch_size)]
                self.files += image_list
        else:
            file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
            self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
            self.files = [self.files[i:i+self.batch_size] for i in range(0, len(self.files), self.batch_size)]

    def _load_data(self, path):
        """
            Load images and Labels
        """
        image_id = path
        image_path = os.path.join(self.image_dir, image_id + '.JPEG')
        # label_path = os.path.join(self.label_dir, image_id + '.png')
        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        # image_id = self.files[index].rsplit("/", -1)[-1].split(".")[0]
        if image_id.find('/') != -1:
            image_id = image_id[image_id.find('/') + 1:]
        if self.prompt_file is not None:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            categories_id = []
            bboxes = []
            scores = []
            if image_id in self.prompts:
                image_prompts = self.prompts[image_id]
                for image_prompt in image_prompts:
                    if image_prompt["score"] > 0.25:
                        scores.append(image_prompt["score"])
                        categories_id.append(image_prompt["category_id"])
                        bboxes.append(image_prompt["bbox"])
            categories_id = np.array(categories_id)
            bboxes = np.array(bboxes)
            if len(bboxes) > 10:
                indices = np.argsort(scores)[::-1]
                categories_id = categories_id[indices]
                bboxes = bboxes[indices]
                categories_id = categories_id[:10]
                bboxes = bboxes[:10]
            return image, None, categories_id, bboxes, image_id
        else:
            return image_path, None, None, None, image_id


class VID(BaseDataLoader):
    def __init__(self, config, prompt_file, conf):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        root = config["dataset"]["data_dir"]
        split = config["dataset"]["split"]
        num_classes = config["dataset"]["num_classes"]
        # conf = config["dataset"]["conf"]

        kwargs = {
            'root': root,
            'split': split,
            'prompt_file': prompt_file,
            'mean': self.MEAN,
            'std': self.STD,
            'batch_size': config["dataloader"]["batch_size"],
            # 'num_classes': config["dataset"]["num_classes"],
            # 'conf': config["dataset"]["conf"],
        }

        if split in ["train", "trainval", "val", "test"]:
            self.dataset = VIDDataset(num_classes, conf, **kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super(VID, self).__init__(self.dataset, config["dataloader"]["batch_size"], config["dataloader"]["num_workers"])



