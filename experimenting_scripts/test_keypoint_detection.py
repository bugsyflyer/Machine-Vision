import cv2
import numpy as np
from matplotlib import pyplot as plt

print("Start loading image")
# Load an image from the media folder
raw_img = cv2.imread("../media/human_follow_2.jpg")
assert raw_img is not None, "file could not be read"
print("Loaded image, starting algorithm")
downscaling_factor = 4 # lower resolution so it goes faster
new_width = raw_img.shape[1] // downscaling_factor
new_height = raw_img.shape[0] // downscaling_factor
image = cv2.resize(raw_img, (new_width, new_height))
print("image resized")

# Create mask around area near object
keypoint_detect_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
# wider circle with lower value corresponding to less keypoints far from the point given
cv2.circle(keypoint_detect_mask, (30 + new_width//16, -90 + new_height//4), 400, 20, thickness=-1)
# smaller circle with high chance of giving keypoints close to the click
cv2.circle(keypoint_detect_mask, (30 + new_width//16, - 90 + new_height//4), 90, 255, thickness=-1)
print(keypoint_detect_mask.shape)
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the ORB detector object
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and compute descriptors using the mask to determine where to sample keypoints
keypoints, descriptors = orb.detectAndCompute(gray, keypoint_detect_mask)

print("Algorithm done")

# Draw keypoints on the image
output_image = cv2.drawKeypoints(image, keypoints, None, color=(255, 255, 0), flags=0)

# Display the original image with keypoints marked
plt.figure(figsize = (10, 8))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
#plt.imshow(keypoint_detect_mask) # show where on image stuff is being selected
plt.title('ORB Feature Detection')
plt.show()
print("done plotting")

print("Starting keypoint matching")

# Load next image
raw_new_img = cv2.imread("../media/human_follow_3.jpg")
assert raw_new_img is not None, "file could not be read"
image_new = cv2.resize(raw_new_img, (new_width, new_height))

gray_new = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)

# Create keypoints on new image
keypoints_2, descriptors_2 = orb.detectAndCompute(gray_new, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
 
# Match descriptors.
matches = bf.match(descriptors,descriptors_2)
 
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(image,keypoints,image_new,keypoints_2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


#find center
list_keypoints_2 = []
list_kp2 = [keypoints_2[item.trainIdx].pt for item in matches[:10]]

points = []

for point in list_kp2:
    points.append([point[0], point[1]])

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


cv2.circle(img3, (image.shape[1]+int(mean_point[0]), int(mean_point[1])), 5, (0,0,255), thickness=4)

plt.imshow(img3),plt.show()
