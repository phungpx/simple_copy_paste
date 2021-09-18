import cv2
import math
import random
import itertools
import numpy as np
import imgaug.augmenters as iaa
from shapely.geometry import Point, box, Polygon

from .base import AleSanityAugmenter
from pathlib import Path

__all__ = ['SimpleCopyPaste']


class SimpleCopyPaste(AleSanityAugmenter):
    def __init__(self, bg_dir, fg_ratio, transforms, region_names, img_pattern='*.*', *args, **kwargs):
        super(SimpleCopyPaste, self).__init__(*args, **kwargs)
        self.bg_dir = bg_dir
        self.bg_imgs = list(Path(bg_dir).glob(img_pattern))
        self.fg_ratio = fg_ratio
        self.transforms = transforms
        self.region_names = region_names

    def resize(self, image, min_size, max_size, resize_if_smaller=False):
        h, w = image.shape[:2]
        min_ratio = min(h, w) / min_size
        max_ratio = max(h, w) / max_size
        ratio = max(min_ratio, max_ratio)
        if resize_if_smaller or ratio > 1:
            image = cv2.resize(image, (0, 0), fx=1 / ratio, fy=1 / ratio)

        return image

    def apply(self, image, label):
        points, json_info = self.get_points(label)

        height, width = image.shape[:2]
        bg_image = random.choice(self.bg_imgs)
        bg_image = cv2.imread(str(bg_image))
        bg_image = self._resize(image=bg_image, min_size=min(width, height),
                                max_size=max(width, height), resize_if_smaller=True)
        bg_image = iaa.PadToFixedSize(width, height)(image=bg_image)
        bg_image = iaa.CropToFixedSize(width, height)(image=bg_image)

        for polygon in polygons:
            # convert polygon to quadangle, and then quadangle to rectangle
            quad = self._poly2quad(polygon=polygon, image_size=(width, height))
            rect = self._quad2rect(quadangle=quad)

            # find radius of hole in image
            area_ratio = random.uniform(*self.hole_area_ratio)
            hole_area = Polygon(rect).area * area_ratio
            radius = int(math.sqrt(hole_area / math.pi))

            # find center_point of hole in image
            M = cv2.getPerspectiveTransform(src=quad, dst=rect)
            array_poly = np.asarray(polygon)
            array_poly = np.concatenate([array_poly, np.ones(shape=(array_poly.shape[0], 1))], axis=1)
            persp_poly = M @ array_poly.T
            persp_poly = persp_poly / persp_poly[-1, :]
            persp_poly = persp_poly[:2, :].T.tolist()

            boundary_ratio = random.uniform(*self.hole_boundary_ratio)
            x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[2][0], rect[2][1]
            center_points = [
                (random.uniform(x1 + radius, x2 - radius),  # top
                 random.uniform(y1 + radius, y1 + boundary_ratio * (y2 - y1) - radius)),
                (random.uniform(x1 + radius, x2 - radius),  # bottom
                 random.uniform(y2 - boundary_ratio * (y2 - y1) + radius, y2 - radius)),
                (random.uniform(x1 + radius, x1 + boundary_ratio * (x2 - x1) - radius),  # right
                 random.uniform(y1 + radius, y2 - radius)),
                (random.uniform(x2 - boundary_ratio * (x2 - x1) + radius, x2 - radius),  # left
                 random.uniform(y1 + radius, y2 - radius))]

            center_points = [point for point in center_points if Polygon(persp_poly).contains(Point(point))]
            center_point = random.choice(center_points)

            # generate hole mask
            M = cv2.getPerspectiveTransform(src=rect, dst=quad)
            mask = self._generate_hole_mask(height=height, width=width, center_point=center_point, radius=radius)
            mask = cv2.warpPerspective(src=mask, M=M, dsize=(width, height))
            ksize = max(max(int(x2), int(y2)) // 500 * 2 + 1, 5)
            mask = cv2.GaussianBlur(mask, ksize=(ksize, ksize), sigmaX=ksize)
            mask = np.expand_dims(mask, axis=-1)

            # combinate image and mask
            image = image * (1. - mask) + bg_image * mask
            image = image.astype(np.uint8)
            augmented = True

        return image, label, augmented