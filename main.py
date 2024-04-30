import cv2 as cv
from benchmark import *
import time
import multiprocessing as mp

if __name__ == "__main__":
    # Test 1 -- Atari Pacman
    source_pacman = cv.imread("imgs/Pacman.png", cv.IMREAD_GRAYSCALE)
    template_pacman = cv.imread("imgs/PacmanPlayer.png", cv.IMREAD_GRAYSCALE)

    # Run tests
    time_linear, returns_linear = benchmark_linear(source_pacman, template_pacman, 1, "ssd")
    time_parallel_4, returns_parallel_4 = benchmark_parallel(source_pacman, template_pacman, 1, 4, "ssd")
    time_parallel_8, returns_parallel_8 = benchmark_parallel(source_pacman, template_pacman, 1, 8, "ssd")
    time_parallel_16, returns_parallel_16 = benchmark_parallel(source_pacman, template_pacman, 1, 16, "ssd")

    # Assert results
    print(returns_linear)
    print(returns_parallel_4)
    print(returns_parallel_8)
    print(returns_parallel_16)
    #assert returns_linear == returns_parallel_4 == returns_parallel_8 == returns_parallel_16, "Results are not the same"

    # Draw and display the detection
    min_coords = returns_linear[0][2]
    cv.rectangle(source_pacman, min_coords, (min_coords[0] + template_pacman.shape[1], min_coords[1] + template_pacman.shape[0]), 255, 2)
    max_coords = returns_linear[0][3]
    max_coords2 = returns_parallel_4[0][3]
    cv.rectangle(source_pacman, max_coords, (max_coords[0] + template_pacman.shape[1], max_coords[1] + template_pacman.shape[0]), 255, 2)
    cv.rectangle(source_pacman, max_coords2, (max_coords2[0] + template_pacman.shape[1], max_coords2[1] + template_pacman.shape[0]), 255, 2)

    cv.imshow("Test 1 -- pacman", source_pacman)

    print("Test 1 -- Atari Pacman")
    print(f"Linear: {time_linear}")
    print(f"Parallel (4): {time_parallel_4}")
    print(f"Parallel (8): {time_parallel_8}")
    print(f"Parallel (16): {time_parallel_16}")

    cv.waitKey(0)
