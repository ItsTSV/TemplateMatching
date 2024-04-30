import multiprocessing as mp
import numpy as np


def match_non_parallel(source: np.ndarray, template: np.ndarray, start_row=None, end_row=None):
    # Get dimensions of source and template
    source_height, source_width = source.shape
    template_height, template_width = template.shape

    # Make sure template is smaller than source
    assert source_height >= template_height and source_width >= template_width, "Template is bigger than source"

    # Match map
    match_map = np.ones_like(source, dtype=np.float32)

    # Start and end rows and cols for processing
    start_row = start_row or 0
    end_row = end_row or source_height - template_height + 1

    # Go through source image
    for y in range(start_row, end_row):
        for x in range(source_width - template_width + 1):
            # Get region (sliding window) of source image
            region = source[y:y + template_height, x:x + template_width]

            # Calculate score based on selected method
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
    # Normalize SSD
    std_region = np.std(region)
    std_template = np.std(template)

    # Prevent zero division
    std_template = max(std_template, 1e-10)
    std_region = max(std_region, 1e-10)

    normalized_ssd = (np.sum((region - template) ** 2) / (std_region * std_template)) / 100

    return normalized_ssd
