import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange
from cumulativeMinimumEnergyMap import cumulative_minimum_energy_map
from find_optimal_vertical_seam import find_optimal_vertical_seam

def reduceWidth(im, energyImage):
    """
    Function to reduce the width of the image by 1 pixel

    :param im: image
    Copying the pixels after the seam.
    :param energyImage: energy image of the image
    :return: image with reduced width
    """
    # get the cumulative minimum energy map
    # get the optimal vertical seam
    # remove the seam from the image
    # return the image with reduced width
    cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, 'VERTICAL')
    seam = find_optimal_vertical_seam(cumulativeEnergyMap)

    #delete the seam from the image from each chanel
    r,c,_ = im.shape
    newImg = np.zeros((r,c,3))
    for i,j in enumerate(seam):
        newImg[i,0:j,:] = im[i,0:j,:] #copy the pixels before the seam
        newImg[i,j:c-1,:] = im[i,j+1:c,:] #copy the pixels after the seam

    newCumulativeEnergyMap = np.zeros((r,c))
    for i,j in enumerate(seam):
        newCumulativeEnergyMap[i,0:j] = cumulativeEnergyMap[i,0:j]
        newCumulativeEnergyMap[i,j:c-1] = cumulativeEnergyMap[i,j+1:c]
    

    return newImg[:,:-1,:].astype(np.uint8), newCumulativeEnergyMap[:,:-1]