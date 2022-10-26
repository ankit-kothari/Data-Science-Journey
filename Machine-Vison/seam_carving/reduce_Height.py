import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange
from find_optimal_horizontal_seam import *
from energyImage import *
from find_optimal_vertical_seam import *
from cumulativeMinimumEnergyMap import cumulative_minimum_energy_map



def reduceHeight(im, energyImage):
    """
    Function to reduce the height of the image by 1 pixel

    :param im: image
    :param energyImage: energy image of the image
    :return: reduced image
    """
    # get the cumulative minimum energy map
    cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, 'HORIZONTAL')
    # get the seam
    seam = find_optimal_horizontal_seam(cumulativeEnergyMap)
    # remove the seam
    # return the image with reduced height
    r,c,_ = im.shape
    newImg = np.zeros((r,c,3))
    for i,j in enumerate(seam):
        newImg[0:j,i,:] = im[0:j,i,:] #copy the first part of the image
        newImg[j:r-1,i,:] = im[j+1:r,i,:] #copy the second part of the image

    newCumulativeEnergyMap = np.zeros((r,c))
    for i,j in enumerate(seam):
        newCumulativeEnergyMap[0:j,i] = cumulativeEnergyMap[0:j,i]
        newCumulativeEnergyMap[j:r-1,i] = cumulativeEnergyMap[j+1:r,i]

    return newImg[:-1,:,:].astype(np.uint8), newCumulativeEnergyMap[:-1,:]
