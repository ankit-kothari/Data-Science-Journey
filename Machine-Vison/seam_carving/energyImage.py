
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange


from scipy.ndimage import sobel, gaussian_filter1d, convolve1d, prewitt
from scipy.linalg import norm
import numpy as np

def energy_image(img):
    '''
    Compute the energy image of an image
    :param img: input image dimensions (MxNx3) of type uint8
    :return: energy image (MxN) of type double
    '''
    
    filter_dx = np.array([[0.25,0.5,0.25],
                   [0,0,0],
                   [-0.25,-0.5,-0.25]])

    filter_dx = np.stack([filter_dx] * 3, axis=2)
    # The folloing filter is used to compute the gradient in the y direction
    filter_dy = np.array([[0.25,0,-0.25],
                   [0.5,0,-0.5],
                   [0.25,0,-0.25]])

    filter_dy = np.stack([filter_dy] * 3, axis=2)

    img = img.astype('double')
    energy = np.absolute(convolve(img, filter_dx)) + np.absolute(convolve(img, filter_dy))
    energy_map = np.sum(energy, axis=2)
    return energy_map.astype('double')
