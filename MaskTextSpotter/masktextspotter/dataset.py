from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
from projects.MaskTextSpotter.masktextspotter.convert_dataset import ScutDataset


def get_gf_dicts(data_dir):
    dataset_dicts = []
    list_filepath = os.path.abspath(os.path.join(data_dir, 'list'))
    with open(list_filepath, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split()
            image_path = os.path.abspath(os.path.join(data_dir, fields[0]))
            label_path = os.path.abspath(os.path.join(data_dir, fields[1]))
            image_label = {'path': image_path, 'boxes': []}
            image_label['boxes'] = ScutDataset(label_path)

            record = {}
            height, width = cv2.imread(image_path).shape[:2]
            record["file_name"] = image_path
            record["height"] = height
            record["width"] = width
            objs = []
            for box in image_label['boxes']:
                obj = {
                    "bbox": [box['xmin'], box['ymin'], box['xmax'], box['ymax']],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

for d in ["trainsets", "testsets"]:
    DatasetCatalog.register("dataset/" + d, lambda d = d: get_gf_dicts("dataset/" + d))
    MetadataCatalog.get("dataset/" + d).set(thing_classes=["ocr"])
tl_metadata = MetadataCatalog.get("dataset/trainsets")