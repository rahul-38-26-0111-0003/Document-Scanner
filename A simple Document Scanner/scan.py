from transform import four_point_transform
from skimage.filters import threshold_adaptive
import numpy as np
import cv2

def resize_height(image, height):
    ratio = height / image.shape[0]
    dim = (int(image.shape[1] * ratio), height)
    res = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return res

def resize_width(image, width):
    ratio = width / image.shape[1]
    dim = (width, int(image.shape[0] * ratio))
    res = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return res

# STEP 1 -> Edge Detection
image = cv2.imread("C:/Users/Rahul/Pictures/mybill.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy() # cloning the image
dim = (int(image.shape[1] * ratio), 500)
resized = resize_height(image, 500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
gray = cv2.GaussianBlur(gray, (5, 5), 0) 
edged = cv2.Canny(gray, 75, 200) 

# Showing the original and the edge detected images
print ("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.imshow("Gray", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# STEP 2 -> Finding Contours

#Finding the contours in the edged image, keeping only the largest ones and initializing the screen contour
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
    # approximating the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4: # considering the case when the approximated contour has 4 points, which is the case when we have found the screen
        screenCnt = approx
        break

"""
Assumptions used here :-
1. The document to be scanned is the main focus of the image.
2. The document is rectangular and thus will have 4 distinct edges.
"""

# showing the contour (outline) of the piece of paper
print ("STEP 2: Finding the contours of the paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# STEP 3 : Applying a perspective transform and threshold

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 251, offset = 10)
warped = warped.astype("uint8") * 255

print ("STEP 3: Applying the Perspective transform")
cv2.imshow("Original", resize_height(orig, 650))
cv2.imshow("Scanned", resize_height(warped, 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
