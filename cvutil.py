import cv2

import numpy as np

from util import PIXEL_SIZE_Y, PIXEL_SIZE_X, si_from_ct


def argwhere(array):
    """ Return something that iterates over (r,c) indices of nonzero pixels

    Same as np.argwhere but returns int32

    >>> import numpy as np, cvutil
    >>> a = np.bool8([[0,1,0],[0,0,1]])
    >>> p = np.argwhere(a)
    >>> print(p)
    [[0 1]
     [1 2]]
    >>> q = cvutil.argwhere(a)
    >>> print(q)
    [[0 1]
     [1 2]]
    >>> print(next(iter(q)))
    [0 1]
    >>>


    :param array:
    :return:
    """

    if array.itemsize == 1:
        array = array.view(np.uint8)
    else:
        array = np.array(array, np.uint8, copy=False)

    return cv2.findNonZero(array).squeeze()[:, ::-1]


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
    image /= cv2.norm(image) / np.sqrt(image.size)
    return image

def make_log(image, sigma):
    image = np.float32(image)
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    image = cv2.Laplacian(image, cv2.CV_32F, borderType=cv2.BORDER_REPLICATE)
    image /= cv2.norm(image) / np.sqrt(image.size)
    return image

def max_filter(image, neighborhood=1, elliptical=True):
    if elliptical:
        footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * neighborhood + 1, 2 * neighborhood + 1))
    else:
        footprint = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * neighborhood + 1, 2 * neighborhood + 1))

    return cv2.dilate(image, footprint, borderType=cv2.BORDER_REPLICATE)


def draw_line(image_array, slope, intercept, color=255, thickness=2, overwrite=False):
    image_array = np.array(image_array, copy=not overwrite)
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


def imwrite(filename, img):
    return cv2.imwrite(filename, img)


def imread(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def correct_tdi_aspect(tdi_array):

    y, x = tdi_array.shape
    y = int(y * PIXEL_SIZE_Y / PIXEL_SIZE_X)  # shrink y to match x pixel size
    # x = int(x * PIXEL_SIZE_X / PIXEL_SIZE_Y) # grow x to match y pixel size

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(int(x / 2750),) * 2)
    # tdi_array = clahe.apply(tdi_array)

    tdi_array = cv2.resize(tdi_array, dsize=(x, y), interpolation=cv2.INTER_CUBIC)

    return tdi_array


def fit_line_fitline(processed_image_array):
    """
    cv2.fitLine internally uses moments, but may iteratively reweight them for robust fit based on DIST_
    but points must be extracted manually

    :param processed_image_array:
    :return slope, intercept, theta:
    """
    points = np.argwhere(processed_image_array)[:, ::-1]  # swap x,y coords
    # points = cv2.findNonZero(processed_image_array) # TODO: is this equivalent?
    line = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
    centroid = line[2:]
    theta = np.arctan2(line[1], line[0])
    slope, intercept = si_from_ct(centroid, theta,processed_image_array.shape)
    return slope, intercept, theta


def fit_line_moments(processed_image_array):
    """
    Use moments to generate whole-image centroid and angle
    works with grayscale data

    :param processed_image_array:
    :return slope, intercept, theta:
    """

    moments = cv2.moments(processed_image_array, True)

    m00 = moments['m00']
    if not m00:
        return 0, processed_image_array.shape[0] / 2, 0

    centroid = np.array((moments['m10'] / m00, moments['m01'] / m00))

    theta = np.arctan2(2 * moments['mu11'], (moments['mu20'] - moments['mu02'])) / 2
    slope, intercept = si_from_ct(centroid, theta, processed_image_array.shape)
    return slope, intercept, theta


def rotate_fiber(image, vshift, theta, overwrite=False):
    image = np.array(image, copy=not overwrite)
    y_size, x_size = image.shape
    mean = int(image.mean())
    translation = np.float32([[1, 0, 0], [0, 1, vshift]])
    # aspect = 2.2894
    # stretch = np.float32([[aspect, 0, 0], [0, 1, 0]])
    # translation = _compose(translation,stretch)
    # theta = np.atan(2*np.tan(theta))
    theta_deg = theta * 180 / np.pi

    # rotation = cv2.getRotationMatrix2D((x_size // 2, y_size // 2), theta_deg, 1)
    # transform = _compose(rotation, translation)
    # return cv2.warpAffine(image, transform, (x_size, y_size), borderValue=mean)

    x_size = image.shape[1]
    max_chunk = 2 ** 15
    patchwidth = 20
    chunks = x_size // max_chunk
    image_parts = []
    for chunk in range(chunks):
        x0 = max_chunk * chunk
        x1 = x0 + max_chunk
        rotation = cv2.getRotationMatrix2D((x_size // 2 - x0, y_size // 2), theta_deg, 1)
        transform = _compose(rotation, translation)
        image_parts.append(cv2.warpAffine(image[:, x0:x1], transform, (max_chunk, y_size),
                                          # borderMode=cv2.BORDER_REPLICATE))
                                          # borderMode=cv2.BORDER_WRAP))
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=mean))

    if x_size % max_chunk:
        x0 = max_chunk * chunks
        rotation = cv2.getRotationMatrix2D((x_size // 2 - x0, y_size // 2), theta_deg, 1)
        transform = _compose(rotation, translation)
        image_parts.append(cv2.warpAffine(image[:, x0:], transform, (x_size - x0, y_size),
                                          # borderMode=cv2.BORDER_REPLICATE))
                                          # borderMode=cv2.BORDER_WRAP))
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=mean))

    output = np.hstack(image_parts)

    for chunk in range(chunks):
        x0 = max_chunk * chunk
        x1 = x0 + max_chunk

        px1 = x1 + patchwidth // 2
        if px1 >= x_size:
            px1 = x_size
        px0 = x1 - patchwidth // 2
        rotation = cv2.getRotationMatrix2D((x_size // 2 - px0, y_size // 2), theta_deg, 1)
        transform = _compose(rotation, translation)
        cv2.warpAffine(image[:, px0:px1], transform, (patchwidth, y_size), dst=output[:, px0:px1],
                       borderMode=cv2.BORDER_REPLICATE)  # borderValue=mean)

    return output


def _compose(a, b):
    return np.dot(np.vstack((a, (0, 0, 1))), np.vstack((b, (0, 0, 1))))[:-1, :]


def binary_threshold(image, threshold):
    return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
