import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from copy import deepcopy
from scipy import sparse
import time
from predict_recommendation import *
from numpy import linalg as LA



def transform2_sparse_matrix(df, rows, cols, column_name="rating"):
    """
    Transforms a dense matrix to a sparse matrix
    """
    return sparse.csc_matrix((df[column_name].values,(df['userId'].values, df['movie_idx'].values)),shape=(rows, cols))

def cost(df, W, U,b,c,global_mean):
    """
    Y: Actual Rating in the compressed row format
    predicted: predicted ratings in the compressed row format
    output: Mean Sqaure Error
    """
    Y = transform2_sparse_matrix(df, W.shape[0], U.shape[0],column_name='rating')
    predicted = transform2_sparse_matrix(predict(df, W, U,b,c,global_mean), W.shape[0], U.shape[0], 'prediction')
    return np.sum((Y-predicted).power(2),dtype=np.float32)/df.shape[0] 
