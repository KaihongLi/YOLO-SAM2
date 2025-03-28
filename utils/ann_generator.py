"""
Created on Wed Mar 19 2025 by LKH
Annotation Generate
"""
import base64
import json
import os
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


# Recursively check for the presence of a bytes type, convert to a utf-8 encoded string
def convert_bytes_to_str(obj):
    if isinstance(obj, dict):
        return {convert_bytes_to_str(key): convert_bytes_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_to_str(element) for element in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    else:
        return obj


def annotation_generator(masks, categories_id, image_id, image_width, image_height, ann_dir: str, classes: list):
    data = dict(version="5.5.0", flags=dict(), shapes=[], imagePath=image_id.split("/")[-1] + '.jpg', imageData=None,
                imageHeight=image_height, imageWidth=image_width)
    shape = dict(label="", points=[], group_id=None, description="", shape_type="mask", flags=dict(),
                 mask="")

    for mask, category_id in zip(masks, categories_id):
        s = deepcopy(shape)
        s["label"] = classes[category_id]
        mask = np.squeeze(mask)
        row_indices, col_indices = np.where(mask == 1)
        y1 = np.min(row_indices)
        x1 = np.min(col_indices)
        y2 = np.max(row_indices)
        x2 = np.max(col_indices)
        s["points"] = [[int(x1), int(y1)], [int(x2), int(y2)]]
        mask = mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8)
        image = Image.fromarray(mask, mode="L")
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        byte_data = buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode('utf-8')
        s["mask"] = base64_str
        data["shapes"].append(s)

    ann_path = os.path.join(ann_dir, image_id + ".json")
    if not os.path.exists(Path(ann_path).parent):
        os.makedirs(Path(ann_path).parent)
    data = convert_bytes_to_str(data)
    with open(ann_path, 'w') as f:
        json.dump(data, f, indent=4)
