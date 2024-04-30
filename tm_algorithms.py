import multiprocessing as mp
import numpy as np


def match_non_parallel(source: np.ndarray, template: np.ndarray, method="ssd", start_row=None, end_row=None) -> tuple:
    # Get dimensions of source and template
    source_height, source_width = source.shape
    template_height, template_width = template.shape

    # SSD -> Lower score, better match; CC -> Higher score, better match
    assert method in ["ssd", "cc"], "Invalid method"

    # Make sure template is smaller than source
    assert source_height >= template_height and source_width >= template_width, "Template is bigger than source"

    # Match map
    match_map = np.zeros((source_height - template_height + 1, source_width - template_width + 1))

    # Start and end rows and cols for processing
    start_row = start_row or 0
    end_row = end_row or source_height - template_height + 1

    # Initialize max/min score and match coordinates
    min_score, max_score = np.inf, -np.inf
    min_coords, max_coords = (0, 0), (0, 0)

    # Go through source image
    for y in range(start_row, end_row):
        for x in range(source_width - template_width + 1):
            # Get region (sliding window) of source image
            region = source[y:y + template_height, x:x + template_width]

            # Calculate score based on selected method
            current_score = calculate_score(region, template, method)

            # Add score to match map
            match_map[y, x] = current_score

            # Update best match
            if current_score <= min_score:
                min_score = current_score
                min_coords = (x, y)

            if current_score >= max_score:
                max_score = current_score
                max_coords = (x, y)

    return min_score, max_score, min_coords, max_coords, match_map


def match_parallel(source: np.ndarray, template: np.ndarray, process_count=4, method="ssd") -> tuple:
    # Get dimensions of source and template
    source_height, source_width = source.shape
    template_height, template_width = template.shape

    # SSD -> Lower score, better match; CC -> Higher score, better match
    assert method in ["ssd", "cc"], "Invalid method"

    # Make sure template is smaller than source
    assert source_height >= template_height and source_width >= template_width, "Template is bigger than source"

    # Initialize max/min score and match coordinates
    min_score, max_score = np.inf, -np.inf
    min_coords, max_coords = (0, 0), (0, 0)

    # Multiprocessing -- Adjust number of processes if it's bigger than number of available CPUs
    process_count = min(process_count, mp.cpu_count())

    # Calculate the number of rows each process will check
    rows_per_process = (source_height - template_height + 1) // process_count

    # Get the start and end rows for each process
    indexes = []
    for i in range(process_count):
        start = i * rows_per_process
        end = start + rows_per_process

        # Make sure the image is checked completely and there's no overflow
        if i == process_count - 1:
            end = source_height - template_height + 1

        indexes.append((start, end))

    # Create a pool of processes
    with mp.Pool(process_count) as pool:
        results = pool.starmap(match_non_parallel, [(source, template, method, index[0], index[1]) for index in indexes])

    # Extract the results from the pool
    match_map = np.zeros((source_height - template_height + 1, source_width - template_width + 1))

    for result in results:
        if result[0] < min_score:
            min_score = result[0]
            min_coords = result[2]

        if result[1] > max_score:
            max_score = result[1]
            max_coords = result[3]

        # Blend match map
        match_map += result[4]

    return min_score, max_score, min_coords, max_coords, match_map


def calculate_score(region, template, method):
    if method == "ssd":
        # Normalize SSD
        std_region = np.std(region)
        std_template = np.std(template)
        normalized_ssd = np.sum((region - template) ** 2) / (std_region * std_template)
        return normalized_ssd
    elif method == "cc":
        # Normalize CC
        std_region = np.std(region)
        std_template = np.std(template)
        cc = np.sum(region * template)
        num_pixels_template = np.prod(template.shape)
        normalized_cc = cc / (std_region * std_template * num_pixels_template)
        return normalized_cc
    else:
        raise ValueError("Invalid method")
