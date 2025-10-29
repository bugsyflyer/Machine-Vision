import cv2
import numpy as np
from matplotlib import pyplot as plt

print("Start loading image")
# Load an image from the media folder
raw_img = cv2.imread("media/tricky_backpack_image.jpg")
assert raw_img is not None, "file could not be read"
print("Loaded image, starting algorithm")
downscaling_factor = 8 # lower resolution so it goes faster
new_width = raw_img.shape[1] // downscaling_factor
new_height = raw_img.shape[0] // downscaling_factor
image = cv2.resize(raw_img, (new_width, new_height))
print("image resized")

# Create mask around area near object
keypoint_detect_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
# wider circle with lower value corresponding to less keypoints far from the point given
cv2.circle(keypoint_detect_mask, (60 + new_width//100, new_height//3), 100, 20, thickness=-1)
# smaller circle with high chance of giving keypoints close to the click
cv2.circle(keypoint_detect_mask, (60 + new_width//100, new_height//3), 20, 255, thickness=-1)
print(keypoint_detect_mask.shape)
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the ORB detector object
orb = cv2.ORB_create(nfeatures=100)

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
