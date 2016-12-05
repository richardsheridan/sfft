from numbers import Number

import cv2
import numpy as np


def sobel_filter(image_array, dx=0, dy=1, ksize=1):
    return cv2.Sobel(image_array, cv2.CV_32F, dx, dy, ksize=ksize,
                     scale=2 ** -(ksize * 2 - 0 - 1 - 2) if ksize > 1 else 1)


def laplacian_filter(image_array, ksize=1):
    # display_pyramid(pyramid)
    log = cv2.Laplacian(image_array, cv2.CV_32F, ksize=ksize, scale=2 ** -(ksize * 2 - 2 - 2 - 2) if ksize > 1 else 1)
    log[0, :] = 0
    log[-1, :] = 0
    return log


def make_pyramid(image, levels=7):
    pyramid = [image]
    x = image
    for i in range(levels):
        x = cv2.pyrDown(x)
        pyramid.append(x)
    return pyramid


def puff_pyramid(pyramid, level, tolevel=0, image=None):
    image = pyramid[level].copy() if image is None else image.copy()
    for i in reversed(range(tolevel, level)):
        image = cv2.pyrUp(image, dstsize=pyramid[i].shape[::-1])
    return image


def display_pyramid(pyramid, cmap='gray'):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(pyramid), 1)
    if len(pyramid) <= 1:
        axs = [axs]
    for im, ax in zip(pyramid, axs):
        ax.imshow(im, cmap=cmap)
    plt.show(1)


def make_log_pyramid(pyramid):
    return [cv2.Laplacian(p, cv2.CV_32F, borderType=cv2.BORDER_REPLICATE) for p in pyramid]


def make_dog_pyramid(pyramid, output_dtype=np.float32, intermediate_dtype=np.float32):
    dogs = [cv2.pyrUp(small, dstsize=big.shape[::-1]).astype(intermediate_dtype) - big.astype(intermediate_dtype)
            for big, small in zip(pyramid, pyramid[1:])]
    if output_dtype != intermediate_dtype:
        dogs = [dog.astype(output_dtype) for dog in dogs]
    return dogs


def make_dog(image, sigma_wide, sigma_narrow):
    if sigma_wide < sigma_narrow:
        sigma_wide, sigma_narrow = sigma_narrow, sigma_wide
    image = np.float32(image)
    return (cv2.GaussianBlur(image, (0, 0), sigma_wide, borderType=cv2.BORDER_REPLICATE) -
            cv2.GaussianBlur(image, (0, 0), sigma_narrow, borderType=cv2.BORDER_REPLICATE))


def make_log(image, sigma):
    image = np.float32(image)
    if sigma > 0:
        blurredimage = cv2.GaussianBlur(np.float32(image), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    else:
        blurredimage = image
    return cv2.Laplacian(blurredimage, cv2.CV_32F, borderType=cv2.BORDER_REPLICATE)


def max_filter(image, neighborhood=1, elliptical=True):
    if elliptical:
        footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * neighborhood + 1, 2 * neighborhood + 1))
    else:
        footprint = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * neighborhood + 1, 2 * neighborhood + 1))

    return cv2.dilate(image, footprint, borderType=cv2.BORDER_REPLICATE)

