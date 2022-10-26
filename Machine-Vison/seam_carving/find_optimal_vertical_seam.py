import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange



def find_optimal_vertical_seam(cumulativeEnergyMap):
    """
    Function to get the horizontal seam in a picture

    :param cumulativeEnergyMap: cumulative energy map of the image
    :return: optimal vertical seam of the image
    
    """
    # backtracking to get the seam
    #print(cumulativeEnergyMap)
    row = len(cumulativeEnergyMap)
    col = len(cumulativeEnergyMap[0])
    seam = [0] * row
    seam[row-1] = np.argmin(cumulativeEnergyMap[row-1])
    for i in range(row-2, -1, -1):
        j = seam[i+1]
        #print(f'j {j}')
        if j == 0:
            seam[i] = np.argmin(cumulativeEnergyMap[i][j:j+2]) + j
        elif j == col - 1:
            #print(f'{cumulativeEnergyMap[i][j-1:j+1]}')
            seam[i] = np.argmin(cumulativeEnergyMap[i][j-1:j+1]) + j-1
        else:
            #print(f'{cumulativeEnergyMap[i][j-1:j+2]}')
            seam[i] = np.argmin(cumulativeEnergyMap[i][j-1:j+2]) + j-1
    return seam