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


def paraboloid(rows, cols, r, c, eccentricity, curvature, theta, ):
    # Eqn of elliptical paraboloid for reference
    # z=(((x-x0)/a)**2+((y-y0)/b)**2)+c
    # TODO: fix rotation

    colshift = (2 * c / (cols - 1) - 1)
    rowshift = (2 * r / (rows - 1) - 1)

    # sin = np.sin(theta)
    # cos = np.cos(theta)
    # colshift = cos*colshift + sin * rowshift
    # rowshift = sin*colshift + cos * rowshift

    i = np.linspace(-1, 1, cols) - colshift
    j = np.linspace(-1, 1, rows) - rowshift
    x = curvature * i
    y = curvature * eccentricity * j
    xx, yy = np.meshgrid(x, y)

    # xx = xx * cos + yy * sin
    # yy = yy * cos + xx * sin

    return xx ** 2 + yy ** 2


def paraboloid_params(max_curvature=1e2, max_eccentricity=1e2):
    return (st.floats(min_value=1 / max_eccentricity,
                      max_value=max_eccentricity,
                      allow_nan=False,
                      ),
            st.floats(min_value=1 / max_curvature,
                      max_value=max_curvature,
                      allow_nan=False,
                      ),
            st.floats(min_value=0,
                      max_value=np.pi / 4,
                      allow_nan=False,
                      ),
            )


def point_of_interest(image_shape):
    rows, cols = image_shape
    return st.tuples(st.floats(min_value=0, max_value=rows - 1), st.floats(min_value=0, max_value=cols - 1))


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


@given(st.integers(min_value=3, max_value=5),
       st.integers(min_value=3, max_value=5),
       st.floats(min_value=0, max_value=4),
       st.floats(min_value=0, max_value=4),
       *paraboloid_params(),
       )
def test_qse_centroid_recovery(rows, cols, r, c, eccentricity, curvature, theta):
    # restrict centroid to within image
    # assume is easier to use than st.data or strategy.flatmap
    assume(0 < r < rows - 1)
    assume(0 < c < cols - 1)

    position = r, c
    image = paraboloid(rows, cols, r, c, eccentricity, curvature, theta)
    min_pixel_loc = min_r, min_c = np.unravel_index(np.argmin(image), (rows, cols))

    # if the min/max pixel is on the border, qse will try to index out of bounds
    assume(0 < min_r < rows - 1)
    assume(0 < min_c < cols - 1)

    # if the paraboloid has two adjacent min/max pixels, it may choose the "wrong" one
    mask = np.ones((rows, cols), dtype=bool)
    mask[min_pixel_loc] = False
    assume(np.all(image[mask] != image[min_pixel_loc]))

    recovered_position = util.quadratic_subpixel_extremum_2d(image, min_pixel_loc)
    assert np.allclose(recovered_position, position, rtol=1e-3, atol=1e-6)


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
