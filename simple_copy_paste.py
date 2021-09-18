import cv2
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from base_copy_paste import BaseCopyPaste

__all__ = ['SimpleCopyPaste']


class SimpleCopyPaste(BaseCopyPaste):
    def __init__(
        self,
        background_dir,
        template_ratio,
        template_type=None,
        image_pattern='*.*',
        background_transforms=None,
        template_transforms=None,
        final_transforms=None,
        *args, **kwargs
    ):
        '''
        Args:
            background_dir: directory where storages all back ground images.
            image_pattern: pattern of all background images.
            template_ratio: ratio between template and background.
            template_type: name of template. Ex: CARD_BACK, CMND_BACK, ...
            transforms: all augmentations applied to template before attaching to background.
        '''
        super(SimpleCopyPaste, self).__init__(*args, **kwargs)
        self.template_type = template_type
        self.template_ratio = template_ratio
        self.final_transforms = final_transforms if final_transforms is not None else []
        self.template_transforms = template_transforms if template_transforms is not None else []
        self.background_transforms = background_transforms if background_transforms is not None else []

        self.bg_images = list(Path(background_dir).glob(image_pattern))

    def apply(self, image, label):
        '''
        Args:
            image: template image
            label: json path or json label of template
        Outputs:
            image: combinated image between template and random background
            label: json label of template on random background
        '''

        image, label, mask, points = self.get_template_info(image, label, self.template_type)

        # choose back ground image randomly
        bg_image = cv2.imread(str(random.choice(self.bg_images)))

        # augment background
        for augmenter in random.sample(
            self.background_transforms,
            k=random.randint(0, len(self.background_transforms))
        ):
            bg_image = augmenter(image=bg_image)

        # augment fore ground template image, teample mask, labeled points of teample
        for augmenter in random.sample(
            self.template_transforms,
            k=random.randint(0, len(self.template_transforms))
        ):
            image, mask, points = augmenter(
                image=image,
                segmentation_maps=mask,
                keypoints=points,
            )

        # set ratio between fore ground and back ground
        height, width = image.shape[:2]
        bg_height, bg_width = bg_image.shape[:2]

        if max(height, width) > min(bg_height, bg_width) * self.template_ratio:
            if height >= width:
                height_new = min(bg_height, bg_width) * self.template_ratio
                width_new = height_new * width / height
            else:
                width_new = min(bg_height, bg_width) * self.template_ratio
                height_new = width_new * height / width

            # resize to appropriate size between foreground and background
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
            bg_width_new, bg_height_new = int(width / self.template_ratio), int(height / self.template_ratio)
            bg_image = iaa.CropToFixedSize(bg_width_new, bg_height_new)(image=bg_image)

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
        k = max(max(image.shape) // 400 * 2 + 1, 5)
        mask = cv2.GaussianBlur(mask, ksize=(k, k), sigmaX=k)

        # combinate fore ground and back ground with fore ground mask
        image = image * mask + bg_image * (1. - mask)
        image = image.astype(np.uint8)

        # apply augmentaions for combinated image
        for augmenter in random.sample(
            self.final_transforms,
            k=random.randint(0, len(self.final_transforms))
        ):
            image = augmenter(image=image)

        return image, label

    def gaussian_shadow(self, image, center=(-1, -1), r=-1):
        '''
        Args:
            image: image: 1, H, W, 3, ndarray
            center: (x,y); (-1,-1) for random
            r: -1 for random
        Outputs:
            shadowed image
        '''
        x_center, y_enter = center
        height, width = image.shape[:2]

        if center == (-1, -1):
            x_center = random.randint(0, width - 1)
            y_center = random.randint(0, height - 1)
        if r == -1:
            r = random.randint(1000, 2000)

        # Generate gaussian
        sigma = r / 3.0
        size = 2 * r
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        mask = np.zeros(shape=(height, width))
        x_min = max(0, x_center - r)
        x_max = min(width, x_center + r)
        y_min = max(0, y_center - r)
        y_max = min(height, y_center + r)

        g_remain = g[(r - (y_center - y_min)):(r + (y_max - y_center)), (r - (x_center - x_min)):(r + (x_max - x_center))]

        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                mask[j, i] = g_remain[j - y_min, i - x_min]

        masks = np.stack([mask, mask, mask], axis=-1)
        masks = np.expand_dims(masks, axis=0)

        # do em agment trên ảnh scan nên em chỉ làm tối đi thôi.
        bright_image = iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.5), add=(-40, 40))(images=image)

        image = bright_image * masks + image * (1 - masks)

        return image.astype(np.uint8)
