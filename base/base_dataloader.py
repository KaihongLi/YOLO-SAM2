"""
    Created on Wed Mar 19 2025 by LKH
    YOLOSAM_DataLoader
"""
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers):
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        self.sampler = YOLOSAMBatchSampler(YOLOSAMSampler(self.dataset), batch_size, drop_last=False)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_sampler': self.sampler,
            'num_workers': num_workers,
            'pin_memory': False,  # cpu
            'collate_fn': collate_fn
        }
        super(BaseDataLoader, self).__init__(**self.init_kwargs)


class YOLOSAMSampler(SequentialSampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.data_source = dataset

    def __iter__(self):
        # random.shuffle(self.data_source.files)
        return iter(self.data_source.files)

    def __len__(self):
        return len(self.data_source)


class YOLOSAMBatchSampler(BatchSampler):
    def __iter__(self):
        for ele in self.sampler:
            yield ele

    def __len__(self):
        return len(self.sampler)


def collate_fn(batch):
    images = []
    labels = []
    categories = []
    bboxes = []
    images_id = []
    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])
        if sample[2] is not None and sample[3] is not None:
            categories.append(sample[2])
            bboxes.append(sample[3])
        images_id.append(sample[4])
    return images, labels, categories, bboxes, images_id


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break
