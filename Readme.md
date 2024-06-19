# Simple Template Matching
This project contains three implementations of Template matching algorithms:

- In tm_algorithms.py, linear and Multiprocessing parallel implementations are located
- In tm_algorithms_numba.py, Numba parallel implementation is located

There are two additional files. Benchmark.py is used to compare the performance of the implementations. Main.py contains
ready-to-run code which loads images, runs all benchmarks and compares and visualizes the results.

The images I used for testing are located in the imgs folder. The Atari ones were taken by me, the Superflat city one
was generated by BingAI, the two paintings were taken from Wikipedia -- they are properly cited in the paper. The 
used thresholds, measured times and Template matching outputs are located in the Results folder (except the Starry Night and
Nighthawks outputs -- they were too big, the email could not be sent with it).

## Dependencies
I used Python 3.10 for this project, but it should work with any Python 3 version. The required libraries are:

- OpenCV
- Numpy
- Multiprocessing
- Numba