import cv2
import numpy
import numpy as np


def arg_min_max(image_array, return_values=False):
    minval, maxval, minpos, maxpos = cv2.minMaxLoc(image_array)

    # fix opencv coordinate conventions
    minpos = minpos[::-1]
    maxpos = maxpos[::-1]

    if return_values:
        return minval, maxval, minpos, maxpos
    else:
        return minpos, maxpos


def sobel_filter(image_array, dx, dy, ksize=1):
    return cv2.Sobel(image_array, cv2.CV_32F, dx, dy, ksize=ksize,
                     scale=2 ** -(ksize * 2 - 0 - 1 - 2) if ksize > 1 else 1)


def laplacian_filter(image_array, ksize=1):
    # display_pyramid(pyramid)
    log = cv2.Laplacian(image_array, cv2.CV_32F, borderType=cv2.BORDER_REPLICATE,
                        ksize=ksize, scale=(2 ** -(ksize * 2 - 2 - 2 - 2) if ksize > 1 else 1),
                        )
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
    image = pyramid[level] if image is None else image
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
    image = (cv2.GaussianBlur(image, (0, 0), sigma_wide, borderType=cv2.BORDER_REPLICATE) -
             cv2.GaussianBlur(image, (0, 0), sigma_narrow, borderType=cv2.BORDER_REPLICATE))
    image /= cv2.meanStdDev(image)[1]
    return image

def make_log(image, sigma):
    image = np.float32(image)
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    image = cv2.Laplacian(image, cv2.CV_32F, borderType=cv2.BORDER_REPLICATE)
    image /= cv2.meanStdDev(image)[1]
    return image

def max_filter(image, neighborhood=1, elliptical=True):
    if elliptical:
        footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * neighborhood + 1, 2 * neighborhood + 1))
    else:
        footprint = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * neighborhood + 1, 2 * neighborhood + 1))

    return cv2.dilate(image, footprint, borderType=cv2.BORDER_REPLICATE)


def draw_line(image_array, slope, intercept, color=255, thickness=2):
    x_size = image_array.shape[1]
    max_chunk = 2 ** 14
    chunks = x_size // max_chunk
    image_parts = []
    for chunk in range(chunks):
        x0 = max_chunk * chunk
        y0 = int(x0 * slope + intercept)
        x1 = max_chunk * (chunk + 1)
        y1 = int(x1 * slope + intercept)
        image_parts.append(cv2.line(image_array[:, x0:x1],
                                    (0, y0),
                                    (max_chunk, y1),
                                    color,
                                    thickness,
                                    ))
    remainder = x_size % max_chunk
    if remainder:
        x0 = max_chunk * chunks
        y0 = int(x0 * slope + intercept)
        x1 = x_size - 1
        y1 = int(x1 * slope + intercept)
        image_parts.append(cv2.line(image_array[:, x0:x1],
                                    (0, y0),
                                    (x1 - x0, y1),
                                    color,
                                    thickness,
                                    ))

    output = np.hstack(image_parts)

    return output