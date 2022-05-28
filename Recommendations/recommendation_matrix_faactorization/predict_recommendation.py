import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from copy import deepcopy
from scipy import sparse


def predict(new_data,W,U,b,c,r_mean):
    '''
    W: Movie Matrix
    U: User Matrix
    b: User Bias (Average rating for all the movies a user rated)
    c: Movie Bias (Averag rating for all the users that rated that movie)
    r_mean: Gloab Avrage of all the ratings in the dataset
    '''
    new_data['prediction'] = np.sum(np.multiply(W[new_data['userId']],U[new_data['movie_idx']]), axis=1,dtype=np.float32)+r_mean+b[new_data['userId']]+c[new_data['movie_idx']]
    new_data['prediction'] = new_data['prediction']
    return new_data
