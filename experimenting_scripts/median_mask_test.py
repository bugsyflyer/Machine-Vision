import numpy as np

points = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [100, 200],
    [4, 5],
])

# median and Median Absolute Deviation
median = np.median(points, axis=0)

mad = np.median(np.abs(points - median), axis=0)
threshold = 3.0

#makes boolean array
mask = np.all(np.abs(points - median) / (mad + 1e-6) < threshold, axis=1)

filtered_points = points[mask]
mean_point = np.mean(filtered_points, axis=0)

print(mean_point)

