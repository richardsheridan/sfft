import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings, assume
from hypothesis.extra.numpy import arrays
from pytest import approx

import sfft.util as util


# MY STRATEGIES


def rnd_len_arrays(dtype, min_len=0, max_len=3, elements=None):
    lengths = st.integers(min_value=min_len, max_value=max_len)
    return lengths.flatmap(lambda n: arrays(dtype, n, elements=elements))


def rnd_shape_images(dtype, min_len=0, max_len=3, elements=None):
    shapes = st.tuples(st.integers(min_value=min_len, max_value=max_len),
                       st.integers(min_value=min_len, max_value=max_len),
                       )
    return shapes.flatmap(lambda n: arrays(dtype, n, elements=elements))


@st.composite
def image_with_peak(draw, min_size=3, max_size=60, max_curvature=1e3, max_eccentricity=1e3):
    rows = draw(st.integers(min_value=min_size, max_value=max_size))
    cols = draw(st.integers(min_value=min_size, max_value=max_size))
    peak_y = draw(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False))
    peak_x = draw(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False))
    eccentricity = draw(
        st.floats(min_value=1 / max_eccentricity, max_value=max_eccentricity, allow_nan=False, allow_infinity=False))
    curvature = draw(
        st.floats(min_value=1 / max_curvature, max_value=max_curvature, allow_nan=False, allow_infinity=False))

    # TODO: introduce a direction for the eccentricity
    # theta = np.pi/2*draw(unit_interval_floats())
    # sin = np.sin(theta)
    # cos = np.cos(theta)
    # z=(((x-x0)/a)**2+((y-y0)/b)**2)+c

    center = (np.array([rows, cols]) - 1) / 2
    peak_position = center * (1 + np.array([peak_y, peak_x]))
    j = np.linspace(-1, 1, rows) - peak_y
    i = np.linspace(-1, 1, cols) - peak_x
    ii, jj = np.meshgrid((curvature * i) ** 2, (curvature * eccentricity * j) ** 2)
    image = ii + jj

    return image, peak_position


def unit_interval_floats(allow_nan=False):
    return st.floats(min_value=0, max_value=1, allow_nan=allow_nan, allow_infinity=False)


### TESTS START HERE




@given(rnd_len_arrays('f8', min_len=100, max_len=1000),
       unit_interval_floats(),
       st.one_of(unit_interval_floats(), st.none()))
@settings(buffer_size=2 ** 16, max_examples=20)  # ,suppress_health_check=[HealthCheck.too_slow])
def test_wavelet_filter(series, sigma, bandwidth):
    # TODO: encode these assertions into the function
    sigma *= len(series) / 9
    if bandwidth is not None:
        bandwidth *= sigma
    with np.errstate(divide='ignore', invalid='ignore'):
        filtered_series = util.wavelet_filter(series, sigma, bandwidth)
    assert len(filtered_series) == len(series)
    assert series.dtype == filtered_series.dtype




@given(rnd_len_arrays('f8', min_len=100, max_len=1000),
       st.one_of(st.sampled_from(util.VALID_ZERO_CROSSING_DIRECTIONS),
                 st.text(min_size=3, max_size=20)))
@settings(buffer_size=2 ** 16, max_examples=40)  # ,suppress_health_check=[HealthCheck.too_slow])
def test_find_zero_crossings(series, direction):
    try:
        with  np.errstate(invalid='ignore', over='ignore'):
            candidates = util.find_zero_crossings(series, direction)
    except ValueError as e:
        message, bad_direction = e.args
        assert message == 'Invalid choice of direction:'
        assert bad_direction not in util.VALID_ZERO_CROSSING_DIRECTIONS
    else:
        assert candidates.dtype == np.bool
        assert len(candidates) == len(series)
        values = series[candidates]


from sfft.util import path_with_stab
import os


@given(st.text(min_size=1), st.sampled_from(util.VALID_IMAGE_EXTENSIONS))
def test_path_with_stab(path, ext):
    assume('.' not in path)
    assume(os.path.basename(path))
    path += ext
    orig_folder, orig_filename = os.path.split(path)
    output = path_with_stab(path)
    folder, filename = os.path.split(output)
    assert filename.startswith(util.STABILIZE_PREFIX)
    assert orig_folder == folder
    if util.STABILIZE_PREFIX in orig_filename:
        assert path == output


@given(st.text(min_size=1), st.sampled_from(util.VALID_IMAGE_EXTENSIONS))
def test_basename_without_stab(path, ext):
    assume('.' not in path)
    assume(os.path.basename(path))
    path += ext
    image_path_with_stab = path_with_stab(path)
    assert util.basename_without_stab(image_path_with_stab) == os.path.splitext(os.path.basename(path))[0]


@given(st.text(min_size=1), st.sampled_from(util.VALID_IMAGE_EXTENSIONS))
def test_stab_roundtrip(path, ext):
    assume('.' not in path)
    assume(os.path.basename(path))
    path += ext
    dirname = os.path.dirname(path)
    assert util.basename_without_stab(path_with_stab(path)) == os.path.splitext(os.path.basename(path))[0]
    path = path_with_stab(path)
    assert os.path.join(dirname, path_with_stab(util.basename_without_stab(path) + ext)) == path




@given(st.one_of(rnd_shape_images('f8', min_len=3, max_len=20),
                 rnd_shape_images('i4', min_len=3, max_len=20)),
       unit_interval_floats(),
       unit_interval_floats())
def test_quadratic_subpixel_extremum(image, a, b):
    assume(a < 1.0)
    assume(b < 1.0)
    r, c = image.shape
    r = int((r - 2) * a) + 1
    c = int((c - 2) * b) + 1
    output = util.quadratic_subpixel_extremum_2d(image, (r, c))
    assert np.linalg.norm(output - np.array([r, c]), np.inf) < 1


@given(image_with_peak())
def test_qse_centroid_recovery(img_pos):
    image, position = img_pos
    r, c = image.shape
    min_pixel_loc = min_r, min_c = np.unravel_index(np.argmin(image), (r, c))
    assume(0 < min_r < r - 1)
    assume(0 < min_c < c - 1)

    assert np.allclose(util.quadratic_subpixel_extremum_2d(image, min_pixel_loc), position, rtol=1e-3, atol=1e-6)


@given(st.one_of(rnd_len_arrays('f8', min_len=3, max_len=10),
                 rnd_len_arrays('f4', min_len=3, max_len=10)),
       unit_interval_floats())
def test_qse_1d(array, index):
    index = int((len(array) - 2) * index) + 1
    output = util.quadratic_subpixel_extremum_1d(array, index)
    assert abs(output - index) < 1


@given(st.integers(min_value=5, max_value=100),
       unit_interval_floats(),
       st.floats(min_value=1e-5, max_value=1e5))
def test_qse_1d_centroid_recovery(a, b, c):
    array_center = (a - 1) / 2
    peak_shift = (b - .5)
    array = c * (np.linspace(-1, 1, a) - peak_shift)
    array *= array
    index = np.argmin(array)
    assert approx((peak_shift + 1) * array_center) == util.quadratic_subpixel_extremum_1d(array, index)
