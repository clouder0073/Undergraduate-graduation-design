from skimage.metrics import structural_similarity
import imutils
import cv2
from matplotlib import pyplot as plt
import numpy as np

imageA = cv2.imread('0.png')#input the origin image
imageB = cv2.imread('1.png')#input the faulty image

graA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
graB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)#turn BGR colorful images into grey images

kernel = np.ones((5,5),np.float32)/500#customise the convolution kernel

grayA = cv2.filter2D(graA,-1,kernel)
grayB = cv2.filter2D(graB,-1,kernel)

(score, diff) = structural_similarity(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM:{}".format(score))#compute and print SSIM

thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]#cv2.threshold means turning values above 127 to 25
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#find the contours of different areas
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)#circle the different areas with rectangles

plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)), plt.title('Origin'), plt.xticks([]), plt.yticks(
        [])
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)), plt.title('Fault location'), plt.xticks([]), plt.yticks(
        []), plt.yticks([])
plt.show()

