import numpy as np
import time
from tm_algorithms import match_non_parallel, match_parallel


def benchmark_linear(source, template, iterations):
    # List to store times and match map (only one needed, the results will be same for every iteration)
    times = []
    match_map = np.ones_like(source, dtype=np.float32)

    for i in range(iterations):
        # Start timer, run the algorithm and stop timer
        start = time.time()
        match_map = match_non_parallel(source, template)
        end = time.time()
        times.append(end - start)

    # Return mean time over all iterations and resulting match map
    return np.mean(times), match_map


def benchmark_parallel(source, template, iterations, process_count):
    # List to store times and match map (only one needed, the results will be same for every iteration)
    times = []
    match_map = np.ones_like(source, dtype=np.float32)

    for i in range(iterations):
        # Start timer, run the algorithm and stop timer
        start = time.time()
        match_map = match_parallel(source, template, process_count)
        end = time.time()
        times.append(end - start)

    # Return mean time over all iterations and resulting match map
    return np.mean(times), match_map
