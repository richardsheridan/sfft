import numpy as np
from pytest import raises, approx
from hypothesis import given, settings, reject, HealthCheck, assume, example
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays


def rnd_len_arrays(dtype, min_len=0, max_len=3, elements=None):
    lengths = st.integers(min_value=min_len, max_value=max_len)
    return lengths.flatmap(lambda n: arrays(dtype, n, elements=elements))


def rnd_shape_images(dtype, min_len=0, max_len=3, elements=None):
    shapes = st.tuples(st.integers(min_value=min_len, max_value=max_len),
                       st.integers(min_value=min_len, max_value=max_len),
                       )
    return shapes.flatmap(lambda n: arrays(dtype, n, elements=elements))


def unit_interval_floats(allow_nan=False):
    return st.floats(min_value=0, max_value=1, allow_nan=allow_nan, allow_infinity=False)


from util import wavelet_filter, STABILIZE_PREFIX


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
        filtered_series = wavelet_filter(series, sigma, bandwidth)
    assert len(filtered_series) == len(series)
    assert series.dtype == filtered_series.dtype


from util import find_zero_crossings, VALID_ZERO_CROSSING_DIRECTIONS


@given(rnd_len_arrays('f8', min_len=100, max_len=1000),
       st.one_of(st.sampled_from(VALID_ZERO_CROSSING_DIRECTIONS),
                 st.text(min_size=3, max_size=20)))
@settings(buffer_size=2 ** 16, max_examples=40)  # ,suppress_health_check=[HealthCheck.too_slow])
def test_find_zero_crossings(series, direction):
    try:
        with  np.errstate(invalid='ignore', over='ignore'):
            candidates = find_zero_crossings(series, direction)
    except ValueError as e:
        message, bad_direction = e.args
        assert message == 'Invalid choice of direction:'
        assert bad_direction not in VALID_ZERO_CROSSING_DIRECTIONS
    else:
        assert candidates.dtype == np.bool
        assert len(candidates) == len(series)
        values = series[candidates]


from util import path_with_stab
import os

from util import VALID_IMAGE_EXTENSIONS


@given(st.text(min_size=1), st.sampled_from(VALID_IMAGE_EXTENSIONS))
def test_path_with_stab(path, ext):
    assume('.' not in path)
    assume(os.path.basename(path))
    path += ext
    orig_folder, orig_filename = os.path.split(path)
    output = path_with_stab(path)
    folder, filename = os.path.split(output)
    assert filename.startswith(STABILIZE_PREFIX)
    assert orig_folder == folder
    if STABILIZE_PREFIX in orig_filename:
        assert path == output


from util import basename_without_stab


@given(st.text(min_size=1), st.sampled_from(VALID_IMAGE_EXTENSIONS))
def test_basename_without_stab(path, ext):
    assume('.' not in path)
    assume(os.path.basename(path))
    path += ext
    image_path_with_stab = path_with_stab(path)
    assert basename_without_stab(image_path_with_stab) == os.path.splitext(os.path.basename(path))[0]


@given(st.text(min_size=1), st.sampled_from(VALID_IMAGE_EXTENSIONS))
def test_stab_roundtrip(path, ext):
    assume('.' not in path)
    assume(os.path.basename(path))
    path += ext
    dirname = os.path.dirname(path)
    assert basename_without_stab(path_with_stab(path)) == os.path.splitext(os.path.basename(path))[0]
    path = path_with_stab(path)
    assert os.path.join(dirname, path_with_stab(basename_without_stab(path) + ext)) == path


from util import quadratic_subpixel_extremum_2d


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
    output=quadratic_subpixel_extremum_2d(image, (r, c))
    assert np.linalg.norm(output-np.array([r,c]),np.inf) < 1


@given(st.integers(min_value=3, max_value=60),
       st.integers(min_value=3, max_value=60),
       st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
       st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
       st.floats(min_value=1e-3, max_value=1e2, allow_nan=False, allow_infinity=False),
       st.floats(min_value=1e-3, max_value=1e2, allow_nan=False, allow_infinity=False))
def test_qse_centroid_recovery(r, c, a, b, f, g):
    center = (np.array([r, c]) - 1) / 2
    position = center * (1 + np.array([a, b]))
    j = np.linspace(-1, 1, r) - a
    i = np.linspace(-1, 1, c) - b
    ii, jj = np.meshgrid(f * i, g * j)
    image = ii ** 2 + jj ** 2
    min_pixel_loc = min_r, min_c = np.unravel_index(np.argmin(image), (r, c))
    assume(0 < min_r < r - 1)
    assume(0 < min_c < c - 1)

    assert np.all(np.isclose(quadratic_subpixel_extremum_2d(image, min_pixel_loc), position, rtol=1e-3, atol=1e-6))
