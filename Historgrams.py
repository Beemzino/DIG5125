import cv2
import matplotlib.pyplot as plt 
import numpy as np 

image = cv2.imread('mario.jpg', 0)

hist = cv2.calcHist([image], [0], None, [256], [0,256])

x_values = np.arrange(256)

plt.bar(x_values, hist.rave1(), color='gray')
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()
