import cv2
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from .base_copy_paste import BaseCopyPaste

__all__ = ['SimpleCopyPaste']


class SimpleCopyPaste(BaseCopyPaste):
    def __init__(self, bg_dir, fg_ratio, transforms, template_type, image_pattern='*.*', *args, **kwargs):
        super(SimpleCopyPaste, self).__init__(*args, **kwargs)
        self.bg_dir = bg_dir
        self.fg_ratio = fg_ratio
        self.transforms = transforms
        self.template_type = template_type

        self.bg_images = list(Path(bg_dir).glob(image_pattern))

    def apply(self, image, label):
        image, label, mask, points = self.get_template_info(image, label)

        # choose back ground image randomly
        height, width = image.shape[:2]
        bg_image = random.choice(self.bg_images)
        bg_image = cv2.imread(str(bg_image))
        bg_image = self._resize(image=bg_image, min_size=min(width, height),
                                max_size=max(width, height), resize_if_smaller=True)
        bg_image = iaa.PadToFixedSize(width, height)(image=bg_image)
        bg_image = iaa.CropToFixedSize(width, height)(image=bg_image)

        # set ratio between fore ground and back ground
        height, width = image.shape[:2]
        bg_height, bg_width = bg_image.shape[:2]

        if max(height, width) > min(bg_height, bg_width) * self.fg_ratio:
            if height >= width:
                height_new = min(bg_height, bg_width) * self.fg_ratio
                width_new = height_new * width / height
            else:
                width_new = min(bg_height, bg_width) * self.fg_ratio
                height_new = width_new * height / width
            # resize
            resize_to_size = iaa.Resize(
                {
                    "height": int(height_new),
                    "width": int(width_new)
                }
            )
            image, mask, points = resize_to_size(
                image=image,
                segmentation_maps=mask,
                keypoints=points
            )
        else:
            bg_width_new, bg_height_new = int(width / self.fg_ratio), int(height / self.fg_ratio)
            bg_image = iaa.CropToFixedSize(bg_width_new, bg_height_new)(image=bg_image)

        # augment fore ground template image, teample mask, labeled points of teample
        for augmenter in self.transforms:
            image, mask, points = augmenter(
                image=image,
                segmentation_maps=mask,
                keypoints=points,
            )

        # pad fore ground image, mask, labeled points to fix with back ground size
        pad_to_size = iaa.PadToFixedSize(
            width=bg_image.shape[1],
            height=bg_image.shape[0]
        )

        image, mask, points = pad_to_size(
            image=image,
            segmentation_maps=mask,
            keypoints=points
        )

        # set all transformed points to new label (json info)
        points = [[float(point.x), float(point.y)] for point in points.keypoints]
        label = self.set_points(label, points)

        # blend fore ground mask with gaussian blur
        mask = mask.get_arr().astype(np.float32)
        ksize = max(max(image.shape) // 400 * 2 + 1, 5)
        mask = cv2.GaussianBlur(mask, ksize=(ksize, ksize), sigmaX=ksize)

        # combinate fore ground and back ground with fore ground mask
        image = image * mask + bg_image * (1. - mask)
        image = image.astype(np.uint8)

        # apply augmentaions for combinated image
        # image = iaa.ChangeColorTemperature(kelvin=11000)(image=image)

        return image, label
