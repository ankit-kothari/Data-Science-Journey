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


def cumulative_minimum_energy_map(energyImage, seamDirection):
    """
    Function to compute the cumulative minimum energy map of an image
    :param energyImage: energy image of the image
    :param seamDirection: string indicating the direction of seam to be removed

    :return: cumulative minimum energy map of the image
    """
    row = energyImage.shape[0]
    col = energyImage.shape[1]
    dp_energy = []

    if seamDirection == 'VERTICAL':
        dp_energy.append(energyImage[0].tolist())

        for i in range(1, row):
            temp = []
            for j in range(col):
                if j == 0:
                    temp.append(energyImage[i][j] + min(dp_energy[i-1][j], dp_energy[i-1][j+1]))
                elif j == col - 1:
                    temp.append(energyImage[i][j] + min(dp_energy[i-1][j], dp_energy[i-1][j-1]))
                else:
                    temp.append(energyImage[i][j] + min(dp_energy[i-1][j-1], dp_energy[i-1][j], dp_energy[i-1][j+1]))
            dp_energy.append(temp)
        dp_return = np.asarray(dp_energy, dtype=np.double)


    if seamDirection == 'HORIZONTAL':
        dp_energy.append(energyImage[:,0].tolist())

        for i in range(1, col):
            temp = []
            for j in range(row):
                if j == 0:
                    temp.append(energyImage[j][i] + min(dp_energy[i-1][j], dp_energy[i-1][j+1]))
                elif j == row - 1:
                    temp.append(energyImage[j][i] + min(dp_energy[i-1][j], dp_energy[i-1][j-1]))
                else:
                    temp.append(energyImage[j][i] + min(dp_energy[i-1][j-1], dp_energy[i-1][j], dp_energy[i-1][j+1]))
            dp_energy.append(temp)
        dp_return = np.asarray(dp_energy, dtype=np.double).T
    return dp_return
