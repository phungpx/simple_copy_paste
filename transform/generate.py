import cv2
import utils
import numpy as np
from pathlib import Path


class Generator():
    def __init__(self, config_path=utils.Path(__file__).with_name('config.yaml')):
        super(Generator, self).__init__()
        config = utils.load_yaml(config_path)
        self.transformer = utils.create_instance(config['transform'])
        self.arg = config['transform_dict']
        self.suffixes = config['suffixes']
        self.mask_suffix = config['mask_suffix']
        self.image_suffix = config['image_suffix']
        self.background_path = config['background_path']
        self.foreground_path = config['foreground_path']
        self.output_path = config['output_path']
        self.sample_per_background = config['sample_per_background']

    def _get_list_image_path(self, input_dir):
        image_paths = []
        for suffix in self.suffixes:
            image_paths.extend(list(Path(input_dir).glob(f'*.{suffix}')))
        return image_paths

    def _transform(self, image, mask):
        if self.arg.get('blur', ''):
            image = self.transformer.blur(image, tuple(self.arg['blur']))

        if self.arg.get('contrast', ''):
            contrast = np.random.randint(self.arg['contrast'][0], self.arg['contrast'][1])
            image = self.transformer.contrast(image, contrast)

        if self.arg.get('noise', ''):
            noise = np.random.randint(self.arg['noise'][0], self.arg['noise'][1])
            image = self.transformer.noise(image, noise)

        if self.arg.get('brightness', ''):
            brightness = np.random.randint(self.arg['brightness'][0], self.arg['brightness'][1])
            image = self.transformer.brightness(image, brightness)

        if self.arg.get('flip', ''):
            flip_mode = self.arg['flip']
            image = self.transformer.flip(image, flip_mode)
            mask = self.transformer.flip(mask, flip_mode)

        if self.arg.get('rotate3D', ''):
            theta = self.arg['rotate3D'].get('theta', 0)
            theta = np.random.uniform(theta[0], theta[1])
            phi = self.arg['rotate3D'].get('phi', 0)
            phi = np.random.uniform(phi[0], phi[1])
            gamma = self.arg['rotate3D'].get('gamma', 0)
            gamma = np.random.uniform(gamma[0], gamma[1])
            dx = self.arg['rotate3D'].get('dx', 0)
            dy = self.arg['rotate3D'].get('dy', 0)
            dz = self.arg['rotate3D'].get('dz', 0)
            image = self.transformer.rotate3D(image, theta, phi, gamma, dx, dy, dz)
            mask = self.transformer.rotate3D(mask, theta, phi, gamma, dx, dy, dz)

        if self.arg.get('rotate2D', ''):
            angle = self.arg['rotate2D'].get('angle', 0)
            angle = np.random.uniform(angle[0], angle[1])
            scale = self.arg['rotate2D'].get('scale', 1)
            scale = np.random.uniform(scale[0], scale[1])
            image = self.transformer.rotate2D(image, angle, None, scale)
            mask = self.transformer.rotate2D(mask, angle, None, scale)

        return image, mask

    def generate_data(self, foreground_path, background_path):
        foreground = cv2.imread(str(foreground_path), cv2.IMREAD_UNCHANGED)
        background = cv2.imread(str(background_path))

        # extract foreground image and foreground mask
        if foreground.shape[2] == 4:
            foreground_image = foreground[:, :, :3]
            foreground_mask = foreground[:, :, 3]
        else:
            foreground_image = foreground
            foreground_mask = np.ones(shape=foreground.shape[:2], dtype=np.uint8) * 255

        # Transform for image and mask
        foreground_image, foreground_mask = self._transform(foreground_image, foreground_mask)

        # Adjust ratio between foreground and background-------------------
        ratio = np.random.uniform(float(self.arg['ratio_hf_per_hb'][0]), float(self.arg['ratio_hf_per_hb'][1]))

        foreground_image = self.transformer.resize(foreground_image, height=int(background.shape[0] * ratio))
        foreground_mask = self.transformer.resize(foreground_mask, height=int(background.shape[0] * ratio))
        # ----------------------------------------------------------------
        hf, wf = foreground_mask.shape[:2]
        hb, wb = background.shape[:2]

        if hb <= hf + 1 or wb <= wf + 1:
            return None, None

        # position of forground_image in background
        # random center of foreground_image from (wf/2 + 1) to (wb - wf/2 + 1)
        x_center = np.random.uniform(0.5 * wf + 1, wb - 0.5 * wf - 1)
        y_center = np.random.uniform(0.5 * hf + 1, hb - 0.5 * hf - 1)
        x1, x2 = int(x_center - 0.5 * wf), int(x_center + 0.5 * wf)
        y1, y2 = int(y_center - 0.5 * hf), int(y_center + 0.5 * hf)

        # foreground_mask: shape(h, w) --> foreground_mask: shape(h, w, 3)
        foreground_mask = np.stack([foreground_mask] * 3, axis=2).astype(np.float32) / 255
        # combined_mask: shape(h, w, c) create a mask for background
        combined_mask = np.zeros_like(background, dtype=np.float32)
        # add foreground_mask into combined_mask to generate a combined_mask
        combined_mask[y1:y2, x1:x2] += foreground_mask

        # remove background area for foreground --> foreground_image final
        foreground_image = foreground_image.astype(np.float32) * foreground_mask
        # remove foreground_image area in background
        combined_image = background.astype(np.float32) * (1 - combined_mask)
        # add foreground_image into background to generate new image
        combined_image[y1:y2, x1:x2] += foreground_image

        # convert float32 type to uint8 type for image
        combined_image = combined_image.astype(np.uint8)
        combined_mask = combined_mask.astype(np.uint8) * 255

        return combined_image, combined_mask

    def main(self):
        output_dir = Path(self.output_path)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        foreground_paths = self._get_list_image_path(self.foreground_path)
        background_paths = self._get_list_image_path(self.background_path)
        for foreground_path in foreground_paths:
            cnt = 0
            for background_path in background_paths:
                for sample in range(int(self.sample_per_background)):
                    combined_image, combined_mask = self.generate_data(foreground_path, background_path)
                    if combined_image is None:
                        continue
                    file_name = foreground_path.stem if cnt == 0 else f'{foreground_path.stem}_{str(cnt)}'
                    cv2.imwrite(str(output_dir.joinpath(file_name + str(self.image_suffix))), combined_image)
                    cv2.imwrite(str(output_dir.joinpath(file_name + str(self.mask_suffix))), combined_mask)
                    print(f'{cnt + 1}th at foreground: {foreground_path.name} and background: {background_path.name}')
                    cnt += 1
        print('__[Finish Processing]__')


if __name__ == '__main__':
    gen = Generator()
    gen.main()
