import multiprocessing as mp
import numpy as np


def match_non_parallel(source: np.ndarray, template: np.ndarray, start_y=None, end_y=None):
    # Get dimensions of source and template
    source_height, source_width = source.shape
    template_height, template_width = template.shape

    # Make sure template is smaller than source
    assert source_height >= template_height and source_width >= template_width, "Template is bigger than source"

    # Match map
    match_map = np.ones_like(source, dtype=np.float32)

    # Start and end rows for processing
    start_y = start_y or 0
    end_y = end_y or source_height - template_height + 1

    # Go through source image
    for y in range(start_y, end_y):
        for x in range(source_width - template_width + 1):
            # Get region (sliding window) of source image
            region = source[y:y + template_height, x:x + template_width]

            # Calculate score (SSD)
            current_score = calculate_score(region, template)

            # Add score to match map
            match_map[y, x] = current_score

    return match_map


def match_parallel(source: np.ndarray, template: np.ndarray, process_count=4):
    # Get dimensions of source and template
    source_height, source_width = source.shape
    template_height, template_width = template.shape

    # Make sure template is smaller than source
    assert source_height >= template_height and source_width >= template_width, "Template is bigger than source"

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
        results = pool.starmap(match_non_parallel, [(source, template, index[0], index[1]) for index in indexes])

    # Extract the results from the pool
    match_map = np.ones_like(source, dtype=np.float32)

    for result in results:
        match_map = np.minimum(match_map, result)

    return match_map


def extract_matches(match_map, threshold):
    # Get the indices of the matches
    y_coords, x_coords = np.where(match_map <= threshold)

    # Zip matches
    matches = list(zip(y_coords, x_coords))

    # Get the number of matches
    num_matches = len(matches)

    return matches, num_matches


def calculate_score(region, template):
    # Calculate the SSD using numpy
    squared_difference = np.square(region - template)
    ssd = np.sum(squared_difference)

    # Normalize the SSD
    normalized_factor = np.prod(template.shape)
    normalized_ssd = ssd / normalized_factor

    return normalized_ssd
