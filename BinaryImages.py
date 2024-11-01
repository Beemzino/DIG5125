import numpy as np
import cv2

ImageA= cv2.imread('lena.tiff', cv2.IMREAD_UNCHANGED)

if ImageA is None:
    print("Error: Could not read the image.")
    exit()

    dims = np.shape(ImageA)

    if len(dims) == 3 and dims [2] == 3: 
        Imagea1 = cv2.cvtColor(ImageA, cv2.COLOR_BGR2GRAY)