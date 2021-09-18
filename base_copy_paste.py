import cv2
import json
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from imgaug.augmentables import Keypoint
from imgaug.augmentables import KeypointsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class BaseCopyPaste(ABC):
    def __call__(self, image, label, template_type):
        image, label = self.apply(image, label, template_type)
        return image, label

    @abstractmethod
    def apply(self, image, label, template_type=None):
        pass

    def get_template(self, image, label, template_type=None) -> Tuple[np.ndarray, SegmentationMapsOnImage, KeypointsOnImage]:
        # get fore ground points from label
        points = self.get_points(label)
        points = [Keypoint(x=point[0], y=point[1]) for point in points]
        points = KeypointsOnImage(keypoints=points, shape=image.shape)

        # get fore ground mask from getting region from label
        mask = self.get_template_mask(label, template_type) if template_type else np.ones_like(image)
        mask = SegmentationMapsOnImage(mask, image.shape)

        return image, mask, points

    def get_template_mask(self, label, template_type: str):
        if isinstance(label, str):
            with open(file=label, mode='r', encoding='utf-8') as f:
                json_info = json.load(f)
        elif isinstance(label, dict):
            json_info = label
        else:
            raise TypeError('label must be str, dict.')

        height, width = json_info['imageHeight'], json_info['imageWidth']
        mask = np.zeros_like(shape=(height, width), dtype=np.uint8)
        is_created = False

        for shape in json_info['shapes']:
            if shape['label'] == template_type:
                if shape['shape_type'] == 'rectangle':
                    points = self.to_4points(shape['points'])
                elif shape['shape_type'] == 'polygon':
                    points = shape['points']
                else:
                    raise ValueError('type of label region must be rectangle or polygon.')

                cv2.fillPoly(img=mask, pts=[np.int32(points)], color=1)

                is_created = True

        if not is_created:
            raise TypeError('label must be contained template label.')

        return mask

    def get_points(self, label):
        if isinstance(label, str):
            with open(file=label, mode='r', encoding='utf-8') as f:
                json_info = json.load(f)
        elif isinstance(label, dict):
            json_info = label
        else:
            raise TypeError('label must be str, dict.')
        points = get_points(json_info)
        return points, json_info

    def set_points(self, json_info, points):
        json_info = set_points(json_info, points)
        return json_info

    def to_4points(points):
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def get_points(json_info):
    points = []
    if isinstance(json_info, dict):
        for key, value in json_info.items():
            if key == 'points':
                points += value
            else:
                points += get_points(value)
    elif isinstance(json_info, list):
        for element in json_info:
            points += get_points(element)

    return points


def set_points(json_info, points):
    if isinstance(json_info, dict):
        for key, value in json_info.items():
            if key == 'points':
                for i in range(len(value)):
                    value[i] = points.pop(0)
            else:
                set_points(value, points)

    elif isinstance(json_info, list):
        for element in json_info:
            set_points(element, points)
