import numba
import numpy as np


# Mostly non-numeric code, no need to use Numba here
def match_numba_parallel(source: np.ndarray, template: np.ndarray, start_row=None, end_row=None):
    # Get dimensions of source and template
    source_height, source_width = source.shape
    template_height, template_width = template.shape

    # Make sure template is smaller than source
    assert source_height >= template_height and source_width >= template_width, "Template is bigger than source"

    # Match map
    match_map = np.ones_like(source, dtype=np.float32)

    # Start and end rows for processing
    start_row = start_row or 0
    end_row = end_row or source_height - template_height + 1

    parallel_outer_loop(source, template, match_map, start_row, end_row)

    return match_map


# Running with nopython and parallel flag; but only the outer loop can be parallelized effectively
@numba.jit(nopython=True, parallel=True)
def parallel_outer_loop(source, template, match_map, start_row, end_row):
    source_height, source_width = source.shape
    template_height, template_width = template.shape

    for y in numba.prange(start_row, end_row):
        for x in range(source_width - template_width + 1):
            region = source[y:y + template_height, x:x + template_width]
            current_score = calculate_score(region, template)
            match_map[y, x] = current_score


@numba.jit(nopython=True)
def calculate_score(region, template):
    # Normalize images
    region = region.astype(np.float32) / 255.0
    template = template.astype(np.float32) / 255.0

    # Calculate SSD
    squared_difference = np.square(region - template)
    ssd = np.sum(squared_difference)

    # Normalize result
    normalized_factor = numba_prod(template.shape)
    return ssd / normalized_factor


# Numba -- for some reason -- throws error while using np.prod, so this is my own implementation
@numba.jit(nopython=True)
def numba_prod(arr):
    result = 1
    for element in arr:
        result *= element

    return result
