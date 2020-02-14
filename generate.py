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

    def _get_list_image_path(self, folder):
        image_paths = []
        paths = [list(Path(folder).glob('*.{}'.format(suffix))) for suffix in self.suffixes]
        for path in paths: image_paths.extend(path)
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
            mask  = self.transformer.rotate3D(mask, theta, phi, gamma, dx, dy, dz)

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
        hf, wf, cf = foreground.shape
        fore_mask  = foreground[:, :, 3] if cf==4 else np.ones((hf, wf), dtype=np.uint8) * 255
        fore_image = foreground[:, :, :3]       
        back_image = cv2.imread(str(background_path))
        hb, wb = back_image.shape[:2]
        # Transform for image and mask
        fore_image, fore_mask = self._transform(fore_image, fore_mask)
        # Adjust ratio between foreground and background-------------------
        ratio = np.random.uniform(float(self.arg['ratio_hf_per_hb'][0]), 
                                float(self.arg['ratio_hf_per_hb'][1]))
        hf = int(hb * ratio)
        fore_image = self.transformer.resize(fore_image, height=hf)
        fore_mask = self.transformer.resize(fore_mask,  height=hf)
        # ----------------------------------------------------------------
        hf, wf, cf = fore_image.shape

        if hb <= hf + 1 or wb <= wf + 1:
            return None, None

        # random center of fore_image from (wf/2 + 1) to (wb - wf/2 + 1)
        x_center = np.random.uniform(0.5 * wf + 1, wb - 0.5 * wf - 1)
        y_center = np.random.uniform(0.5 * hf + 1, hb - 0.5 * hf - 1)
        x1, x2 = int(x_center - 0.5 * wf), int(x_center + 0.5 * wf)
        y1, y2 = int(y_center - 0.5 * hf), int(y_center + 0.5 * hf)
        
        # fore_mask: shape(h, w) --> fore_mask: shape(h, w, 1) 
        # np.newaxis: increase dimension by one more dimension
        fore_mask = fore_mask[:, :, np.newaxis].astype(np.float32) / 255 
        # back_mask: shape(h, w, c) create a mask for background
        back_mask = np.zeros_like(back_image, dtype=np.float32)
        # add fore_mask into back_mask to generate a mask
        back_mask[y1:y2, x1:x2] += fore_mask
        # concate fore_mask * 3 follow axis=2 --> shape(h, w, 3)
        fore_mask = np.concatenate([fore_mask] * 3, axis=-1)
        # remove background area for foreground --> fore_image final
        fore_image = fore_image.astype(np.float32) * fore_mask
        # remove fore_image area in back_image
        back_image = back_image.astype(np.float32) * (1 - back_mask)
        # add fore_image into back_image to generate new image
        back_image[y1:y2, x1:x2] += fore_image
        # convert float32 type to uint8 type for image
        output_image = back_image.astype(np.uint8)
        output_mask = (back_mask * 255).astype(np.uint8)
        
        return output_image, output_mask

    def main(self):
        output_dir = Path(self.output_path)
        foreground_paths = self._get_list_image_path(self.foreground_path)
        background_paths = self._get_list_image_path(self.background_path)
        for fore_path in foreground_paths:
            idx = 0
            for back_path in background_paths:
                for sample in range(int(self.sample_per_background)):
                    image, mask = self.generate_data(fore_path, back_path)
                    if image is None:
                        continue
                    file_name = fore_path.stem if idx==0 else fore_path.stem + '_' + str(idx) 
                    idx += 1
                    cv2.imwrite(str(output_dir.joinpath(file_name + str(self.image_suffix))), image)
                    cv2.imwrite(str(output_dir.joinpath(file_name + str(self.mask_suffix))), mask)
                    print('[INFO {} th] at foreground: {} and background: {}'.format(idx, fore_path.name, back_path.name))
        print('[INFO] finish process')

if __name__ == '__main__':
    gen = Generator()
    gen.main()




