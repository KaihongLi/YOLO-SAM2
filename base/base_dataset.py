"""
    Created on Wed Mar 19 2025 by LKH
    YOLO+SAM2_Dataset
"""
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataSet(Dataset):
    def __init__(self, root, split, prompt_file, mean, std, batch_size=16):
        self.root = root
        self.split = split
        self.prompt_file = prompt_file
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.files = []
        self.prompts = {}
        self._set_files()
        # self.to_tensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean, std)

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, categories_id, bboxes, image_id = self._load_data(index)

        return image, label, categories_id, bboxes, image_id

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
