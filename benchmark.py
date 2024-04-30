import numpy as np
import time
from tm_algorithms import match_non_parallel, match_parallel


def benchmark_linear(source, template, iterations, method):
    times, locations = [], []

    for i in range(iterations):
        # Run template matching
        start = time.time()
        min_val, max_val, min_coords, max_coords, coords_map = match_non_parallel(source, template, method)
        end = time.time()

        # Append data
        times.append(end - start)
        locations.append((min_val, max_val, min_coords, max_coords))

    return np.mean(times), locations


def benchmark_parallel(source, template, iterations, process_count, method):
    times, locations = [], []

    for i in range(iterations):
        # Run template matching
        start = time.time()
        min_val, max_val, min_coords, max_coords = match_parallel(source, template, process_count, method)
        end = time.time()

        # Append data
        times.append(end - start)
        locations.append((min_val, max_val, min_coords, max_coords))

    return np.mean(times), locations
