import numpy as np


def pred_funtion(x,wk):
    prediction = wk.T.dot(x) #returns (1XN)
    #print(prediction.shape)
    #print(prediction)
    return prediction
