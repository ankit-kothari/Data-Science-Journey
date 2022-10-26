import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange
from find_optimal_vertical_seam import *
from find_optimal_horizontal_seam import *
from reduce_Width import *
from reduce_Height import *
from  energyImage import *
from cumulativeMinimumEnergyMap import *

#load the image
#reduce  of the image by 100 pixels
image = cv2.imread('inputSeamCarvingPrague.jpg')
im1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f'Original image size {im1.shape}')
energyImage = energy_image(im1)
for i in range(100):
    im1,_ = reduceHeight(im1, energyImage)
    energyImage = energy_image(im1)
plt.imshow(im1)
print(f'Reduce width of image {im1.shape}')
cv2.imwrite('outputReduceHeightPrague.png', cv2.cvtColor(im1, cv2.COLOR_RGB2BGR))


#reduce width of the image by 100 pixels
image2 = cv2.imread('inputSeamCarvingMall.jpg')
im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
print(f'Original image size {im2.shape}')
energyImage = energy_image(im2)
for i in range(100):
    im2,_ = reduceHeight(im2, energyImage)
    energyImage = energy_image(im2)
plt.imshow(im2)
print(f'Reduce width of image {im2.shape}')
cv2.imwrite('outputReduceHeightMall.png', cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))
