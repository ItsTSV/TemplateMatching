import numpy as np
import time
from tm_algorithms import match_non_parallel, match_parallel
from tm_algorithms_numba import match_numba_parallel


def benchmark(source, template, iterations, method, process_count=1):
    # List to store times and match map (only one needed, the results will be same for every iteration)
    times = []
    match_map = np.ones_like(source, dtype=np.float32)

    for i in range(iterations):
        # Start timer, run the algorithm and stop timer
        start = time.time()
        if method == "non_parallel":
            match_map = match_non_parallel(source, template)
        elif method == "parallel":
            match_map = match_parallel(source, template, process_count)
        elif method == "numba":
            match_map = match_numba_parallel(source, template)
        else:
            raise ValueError("Invalid method")
        end = time.time()
        times.append(end - start)

    # Return mean time over all iterations and resulting match map
    return np.mean(times), match_map

