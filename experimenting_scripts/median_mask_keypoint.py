import numpy as np

keypoints = []

points = []

for point in keypoints:
    points.append[point.x, point.y]

points_array = np.array(points)

# median and Median Absolute Deviation
median = np.median(points_array, axis=0)

mad = np.median(np.abs(points_array - median), axis=0)
threshold = 3.0

#makes boolean array
mask = np.all(np.abs(points_array - median) / (mad + 1e-6) < threshold, axis=1)

filtered_points = points_array[mask]
mean_point = np.mean(filtered_points, axis=0)

print(mean_point)