import cv2 as cv
from tm_algorithms import match_non_parallel, match_parallel
import time

if __name__ == "__main__":
    # Load two images
    source = cv.imread("imgs/source.png", cv.IMREAD_GRAYSCALE)
    template = cv.imread("imgs/template.png", cv.IMREAD_GRAYSCALE)

    # Match using opencv's matchTemplate
    time_start = time.time()
    result = cv.matchTemplate(source, template, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    time_end = time.time()
    print(f"-----------------------------\nOpenCV matchTemplate:\nMin Coords: {min_loc}\nMax Coords: {max_loc}\nTime taken: {time_end - time_start}\n-----------------------------")

    # Match using non-parallel implementation
    time_start = time.time()
    min_val, max_val, min_loc, max_loc = match_non_parallel(source, template, method="ssd")
    time_end = time.time()
    print(f"Non-parallel implementation:\nMin Coords: {min_loc}\nMax Coords: {max_loc}\nTime taken: {time_end - time_start}\n-----------------------------")

    # Match using parallel implementation
    time_start = time.time()
    min_val, max_val, min_loc, max_loc = match_parallel(source, template, process_count=4, method="ssd")
    time_end = time.time()
    print(f"Parallel implementation:\nMin Coords: {min_loc}\nMax Coords: {max_loc}\nTime taken: {time_end - time_start}\n-----------------------------")


