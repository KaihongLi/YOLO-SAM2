"""
    Original from https://github.com/facebookresearch/sam2
    Modified by LKH
"""
import numpy as np
from matplotlib import pyplot as plt


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax, category_id, score, classes: list, palette: list):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # color = [c / 255 for c in palette[category_id % len(palette)]]
    color = [c / 255 for c in palette[category_id*3:category_id*3+3]]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))
    text = f'{classes[category_id]}: {float(np.squeeze(score)) * 100:.2f}'
    txt_color = (0, 0, 0) if np.mean(color) > 0.5 else (1, 1, 1)
    txt_bk_color = [c * 0.7 for c in color]
    # font_dict = {
    #     'family': 'serif',  # 字体系列
    #     'color': 'darkred',  # 字体颜色
    #     'weight': 'normal',  # 字体粗细
    #     'size': 10,  # 字体大小
    # }
    fontsize = 10
    ax.text(x0 + fontsize, y0 - int(1.5 * fontsize), text, fontsize=fontsize, color=txt_color,
            bbox=dict(boxstyle='square,pad=0.3', facecolor=txt_bk_color))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
