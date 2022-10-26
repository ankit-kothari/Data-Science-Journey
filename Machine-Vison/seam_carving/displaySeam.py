import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange



def display_seam(im,seam, type):
    """
    Function to display the seam on the image

    :param im: image
    :param seam: seam to be displayed
    :param type: type of seam to be displayed
    :return: image with seam displayed
    """
    if type == 'VERTICAL':
        for i in range(len(seam)):
            im[i][seam[i]] = [255, 0, 0]
    if type == 'HORIZONTAL':
        for i in range(len(seam)):
            im[seam[i]][i] = [255, 0, 0]

    #return the image with seam displayed

    return plt.imshow(im)
