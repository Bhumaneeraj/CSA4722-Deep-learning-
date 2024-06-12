import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread(r'E:/Deep learning/datasets/image_0.jpg')

# Convert BGR to RGB for displaying
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological closing (dilation followed by erosion)
kernel = np.ones((2, 2), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Perform dilation to get sure background
sure_bg = cv2.dilate(closing, kernel, iterations=3)

# Plot the images
plt.subplot(211), plt.imshow(closing, 'gray')
plt.title("morphologyEx:Closing:2x2"), plt.xticks([]), plt.yticks([])

plt.subplot(212), plt.imshow(sure_bg, 'gray')
plt.title("Dilation"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

# Save the dilation image
cv2.imwrite(r'dilation.png', sure_bg)
