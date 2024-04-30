import cv2 as cv
from benchmark import *
from tm_algorithms import *
import time
import multiprocessing as mp

if __name__ == "__main__":
    # Test 1 -- Atari Pacman
    source_pacman = cv.imread("imgs/SpaceInvaders.png", cv.IMREAD_GRAYSCALE)
    template_pacman = cv.imread("imgs/SpaceInvader.png", cv.IMREAD_GRAYSCALE)

    # Run tests0
    match_map_linear = match_non_parallel(source_pacman, template_pacman)
    match_map_parallel_4 = match_parallel(source_pacman, template_pacman, 4)
    match_map_parallel_8 = match_parallel(source_pacman, template_pacman, 8)
    match_map_parallel_16 = match_parallel(source_pacman, template_pacman, 16)

    # Check if results are the same using a.all
    assert match_map_linear.all() == match_map_parallel_4.all() == match_map_parallel_8.all() == match_map_parallel_16.all(), "Results are not the same"

    # Extract results
    matches, count = extract_matches(match_map_linear, 0.001)

    # Draw results
    color_pacman = cv.cvtColor(source_pacman, cv.COLOR_GRAY2BGR)
    for match in matches:
        y, x = match
        cv.rectangle(color_pacman, (x, y), (x + template_pacman.shape[1], y + template_pacman.shape[0]), (0, 255, 0), 2)

    cv.imshow("Pacman", color_pacman)
    cv.waitKey(0)
    print(f"Number of matches: {count}")