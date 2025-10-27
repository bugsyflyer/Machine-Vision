import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


raw_img = cv.imread('tricky_backpack_image.jpg')
downscaling_factor = 32

assert raw_img is not None, "file could not be read, check with os.path.exists()"
print(raw_img.shape)
print("reducing resolution")
new_width = raw_img.shape[1] // downscaling_factor
new_height = raw_img.shape[0] // downscaling_factor
img = cv.resize(raw_img, (new_width, new_height))
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (2,2,1500//downscaling_factor,1500//downscaling_factor)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,3,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
