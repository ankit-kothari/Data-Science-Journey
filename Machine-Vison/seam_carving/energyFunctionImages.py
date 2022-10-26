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


################################################################
#########################part a ################################
################################################################


image = cv2.imread('inputSeamCarvingPrague.jpg')
im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
energyImage = energy_image(im)

#plot the energy image
plt.imsave('outputPragueEnergy.png', energyImage)
plt.imshow(energyImage)



#####################################################################
#########################part 3b VERTICAL ###########################
#####################################################################

#plot the cumulative energy map
cum_energy_v = cumulative_minimum_energy_map(energyImage, 'VERTICAL')
plt.imshow(cum_energy_v)

#save the cumulative energy map
plt.imsave('outputCumEnergyVertical.png', cum_energy_v)





#####################################################################
#########################part 3b HORIZONTAL ###########################
#####################################################################

#plot the horizontal cumulative energy map
cum_energy_h = cumulative_minimum_energy_map(energyImage, 'HORIZONTAL')
plt.imshow(cum_energy_h)

#save the cumulative energy map
plt.imsave('outputCumEnergyHorizontal.png', cum_energy_h)
