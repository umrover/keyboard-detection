import numpy as np
import cv2 as cv


# See https://medium.com/@itberrios6/how-to-apply-motion-blur-to-images-75b745e3ef17
def get_motion_blur_kernel(theta, thickness=1, ksize=21):
    """ Obtains Motion Blur Kernel

    @param theta - direction of blur
    @param thickness - thickness of blur kernel line
    @param ksize - size of blur kernel
    """
    c = ksize // 2

    theta = np.radians(theta)
    x = int(np.cos(theta) * 100)
    y = int(np.sin(theta) * 100)

    _kernel = np.zeros((ksize, ksize))
    _kernel = cv.line(_kernel, (c + x, c + y), (c, c), (1,), thickness)
    return _kernel / _kernel.sum()


def get_vignette_kernel(sigma, size):
    _kernel_x = cv.getGaussianKernel(size[0], sigma)
    _kernel_y = cv.getGaussianKernel(size[1], sigma)
    _kernel = _kernel_y * _kernel_x.T

    _kernel /= _kernel.max()
    return np.repeat(_kernel[:, :, np.newaxis], 3, axis=2)
