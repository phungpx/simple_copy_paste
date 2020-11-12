import os
import cv2
import numpy as np


class Transform():
    """
        Define some transform function which use to augmentation data
    """
    @staticmethod
    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            dim = (int(w * height / float(h)), height)
        elif height is None:
            dim = (width, int(h * width / float(w)))
        else:
            dim = (width, height)
        return cv2.resize(image, dim, interpolation=inter)

    @staticmethod
    def flip(image, mode=0):
        '''Flip image
        Args:
            image:
            mode:
                0  (vertical flip)
                1  (horizontal flip)
                -1 (vertical and horizontal flip)
        Output:
            fliped image
        '''
        if mode == 0 or mode == 1 or mode == -1:
            fliped = cv2.flip(image, int(mode))
        else:
            fliped = image
        return fliped

    @staticmethod
    def translate(image, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, image.shape[1::-1])
        return shifted

    @staticmethod
    def rotate2D(image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (nW, nH))
        return rotated

    @staticmethod
    def rotate3D(image, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        """Rotate image given angle
        Args:
            image : original image that you want rotate
            gamma (float - degree): rotation angle of z axis (basically a 2D rotation)
            theta (float - degree): rotation angle of x axis
            phi (float - degree): rotation angle of y axis
            dx: translation along the x axis
            dy: translation along the y axis
            dz: translation along the z axis (distance to the image)
        Return:
            A rotated numpy array image
        """
        h, w = image.shape[:2]

        gamma = gamma * np.pi / 180
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180

        # get ideal focal length on z axis
        d = np.sqrt(h ** 2 + w ** 2)
        focal = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)  # Oz axis scale (if < 0 is flip)
        dz = focal

        A1 = np.array([[1, 0, - w / 2],
                       [0, 1, - h / 2],
                       [0, 0, 1],
                       [0, 0, 1]], dtype="float32")

        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta), - np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0],
                       [0, 0, 0, 1]], dtype="float32")

        Ry = np.array([[np.cos(phi), 0, - np.sin(phi), 0],
                       [0, 1, 0, 0],
                       [np.sin(phi), 0, np.cos(phi), 0],
                       [0, 0, 0, 1]], dtype="float32")

        Rz = np.array([[np.cos(gamma), - np.sin(gamma), 0, 0],
                       [np.sin(gamma), np.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype="float32")

        # Translation matrix
        T = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]], dtype="float32")

        A2 = np.array([[focal, 0, w / 2, 0],
                       [0, focal, h / 2, 0],
                       [0, 0, 1, 0]], dtype="float32")

        transfom = A2 @ (T @ ((Rx @ Ry @ Rz) @ A1))
        warped = cv2.warpPerspective(image, transfom, (w, h))
        return warped

    @staticmethod
    def blur(image, ks=(3, 3)):
        """adding blur into image
        Args:
            image (numpy array): original image
            ks (typle): kernel size
        Return:
            blured image
        """
        return cv2.blur(image, ks)

    @staticmethod
    def contrast(image, contrast=30):
        """adjust contrast for input image.
        Args:
            image (numpy array): input image
        Returns:
            output_image (numpy array): image after adjust contrast.
        """
        output_image = None
        if contrast:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha, beta, gamma = f, 0, 127 * (1 - f)
            output_image = cv2.addWeighted(image, alpha, image, beta, gamma)
            # dst = src1 * alpha + src2 * beta + gamma (addWeighted)
        else:
            output_image = image

        return output_image

    @staticmethod
    def brightness(image, brightness=10):
        """
        Adjust brightness for input image.
        Args:
            image (numpy array): input image.
        Returns:
            output_image (numpy array): image after adjust brightness.
        """
        output_image = None
        if brightness:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha, beta, gamma = (highlight - shadow) / 255, 0, shadow
            output_image = cv2.addWeighted(image, alpha, image, beta, gamma)
            # dst = src1 * alpha + src2 * beta + gamma (addWeighted)
        else:
            output_image = image

        return output_image

    @staticmethod
    def noise(image, var=50):
        """
        Add gaussian noise to image
        Args:
            image: original image
            var: variance of distribution
        Return:
            noisy_image: An numpy image with noise
        """
        h, w = image.shape[:2]

        mean = 0
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (h, w))

        noisy_image = np.zeros(image.shape, np.float32)

        if len(image.shape) == 2:
            noisy_image = image + gaussian
        else:
            noisy_image[:, :, 0] = image[:, :, 0] + gaussian
            noisy_image[:, :, 1] = image[:, :, 1] + gaussian
            noisy_image[:, :, 2] = image[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        return noisy_image
