import cv2 as cv
from benchmark import *
from tm_algorithms import extract_matches

if __name__ == "__main__":
    # Prepare images
    source_address = "imgs/SpaceInvaders.png"
    template_address = "imgs/Invader.png"
    source_img = cv.imread(source_address, cv.IMREAD_GRAYSCALE)
    template_img = cv.imread(template_address, cv.IMREAD_GRAYSCALE)

    # Prepare benchmark parameters
    iterations = 1
    threshold = 0.01

    # Run benchmarks
    mean_time_non_parallel, match_map_non_parallel = benchmark(source_img, template_img, iterations, "non_parallel")
    print("Non-parallel done")
    mean_time_parallel2, match_map_parallel2 = benchmark(source_img, template_img, iterations, "parallel", process_count=2)
    print("Parallel 2 done")
    mean_time_parallel4, match_map_parallel4 = benchmark(source_img, template_img, iterations, "parallel", process_count=4)
    print("Parallel 4 done")
    mean_time_parallel8, match_map_parallel8 = benchmark(source_img, template_img, iterations, "parallel", process_count=8)
    print("Parallel 8 done")
    mean_time_numba, match_map_numba = benchmark(source_img, template_img, iterations, "numba")
    print("Numba done\n")

    # Extract the results
    matches_non_parallel, number_of_matches_non_parallel = extract_matches(match_map_non_parallel, threshold=threshold)
    matches_parallel2, number_of_matches_parallel2 = extract_matches(match_map_parallel2, threshold=threshold)
    matches_parallel4, number_of_matches_parallel4 = extract_matches(match_map_parallel4, threshold=threshold)
    matches_parallel8, number_of_matches_parallel8 = extract_matches(match_map_parallel8, threshold=threshold)
    matches_numba, number_of_matches_numba = extract_matches(match_map_numba, threshold=threshold)

    # Assert the results are the same
    assert (number_of_matches_non_parallel == number_of_matches_parallel2 == number_of_matches_parallel4 ==
            number_of_matches_parallel8 == number_of_matches_numba), "Number of matches is not the same"
    assert matches_non_parallel == matches_parallel2 == matches_parallel4 == matches_parallel8 == matches_numba, "Matches are not the same"

    # Print results
    print(f"Found {number_of_matches_non_parallel} matches with threshold {threshold}. Results are the same.")
    print("Mean time non-parallel: {:.3f} s".format(mean_time_non_parallel))
    print("Mean time parallel (2 processes): {:.3f} s".format(mean_time_parallel2))
    print("Mean time parallel (4 processes): {:.3f} s".format(mean_time_parallel4))
    print("Mean time parallel (8 processes): {:.3f} s".format(mean_time_parallel8))
    print("Mean time Numba parallel: {:.3f} s".format(mean_time_numba))

    # Draw and show the image
    color_source = cv.imread(source_address)
    for match in matches_parallel8:
        y, x = match
        cv.rectangle(color_source, (x, y), (x + template_img.shape[1], y + template_img.shape[0]), (0, 255, 0), 2)

    cv.imshow("Matches", color_source)
    cv.waitKey(0)
