"""
    Original from https://github.com/yassouali/pytorch-segmentation
"""
import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


def batch_pix_accuracy(predict, target, labeled):
    """
    Numpy implementation
    Computes the accuracy of a batch of predictions
    """
    # Calculate the total number of pixels in the marked area
    pixel_labeled = labeled.sum()

    # Calculate the number of pixels where the prediction and target match and are labelled
    pixel_correct = ((predict == target) & labeled).sum()

    if pixel_correct > pixel_labeled:
        print(predict.shape, target.shape, labeled.shape)
    # Ensure that the number of correctly predicted pixels does not exceed the number of pixels in the marked area
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"

    # Returns the number of correctly predicted pixels and the number of pixels in the labelled area
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, num_class, labeled):
    """
    Numpy implementation
    Computes the intersection and union of a batch of predictions and a batch of targets
    """

    # Multiplying predict with labeled, using the broadcast mechanism
    predict = predict * labeled

    # Compute the intersection
    intersection = predict * (predict == target)

    # Calculate histogram using np.bincount
    area_inter = np.bincount(intersection.ravel().astype(np.int64), minlength=num_class + 1)[1:]
    area_pred = np.bincount(predict.ravel().astype(np.int64), minlength=num_class + 1)[1:]
    area_lab = np.bincount(target.ravel().astype(np.int64), minlength=num_class + 1)[1:]

    # Compute the union
    area_union = area_pred + area_lab - area_inter

    # Ensure that the intersection area is less than or equal to the union area
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return area_inter, area_union


def batch_pix_accuracy_gpu(predict, target, labeled):
    """
    Pytorch implementation
    """
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union_gpu(predict, target, num_class, labeled):
    """
    Pytorch implementation
    """
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(outputs, targets, num_class):
    # _, predict = torch.max(output.data, 1)
    height = outputs[0].shape[0]
    width = outputs[0].shape[1]
    same = True
    for i in range(len(outputs)):
        if outputs[i].shape[0] != height or outputs[i].shape[1] != width:
            same = False
    if same:
        predict = np.stack(outputs, axis=0) + 1
        target = np.stack(targets, axis=0) + 1

        labeled = (target > 0) * (target <= num_class)
        correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
        inter, union = batch_intersection_union(predict, target, num_class, labeled)
        return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5)]
    else:
        correct_all = 0
        num_labeled_all = 0
        inter_all = 0
        union_all = 0
        for i in range(len(outputs)):
            predict = outputs[i] + 1
            target = targets[i] + 1

            labeled = (target > 0) * (target <= num_class)
            correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
            inter, union = batch_intersection_union(predict, target, num_class, labeled)
            correct_all += correct
            num_labeled_all += num_labeled
            inter_all += inter
            union_all += union

        return [np.round(correct_all, 5), np.round(num_labeled_all, 5), np.round(inter_all, 5), np.round(union_all, 5)]
