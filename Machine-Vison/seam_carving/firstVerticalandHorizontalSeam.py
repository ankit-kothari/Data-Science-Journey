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
from displaySeam import *




################################################################
#################### First Horizontal Seam ######################
#################################################################



image = cv2.imread('inputSeamCarvingPrague.jpg')
im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
energyImage = energy_image(im)
cum_energy = cumulative_minimum_energy_map(energyImage, 'HORIZONTAL')
seam = find_optimal_horizontal_seam(cum_energy)
display_seam(im, seam, 'HORIZONTAL')



################################################################
#################### First VERTICAL Seam ######################
#################################################################

image = cv2.imread('inputSeamCarvingPrague.jpg')
im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
energyImage = energy_image(im)
cum_energy = cumulative_minimum_energy_map(energyImage, 'VERTICAL')
seam = find_optimal_vertical_seam(cum_energy)
display_seam(im, seam, 'VERTICAL')
