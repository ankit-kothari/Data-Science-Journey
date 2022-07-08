import numpy as np
from prediction_function import *

def cost_calc(x,w,y):
    prediction = pred_funtion(x,w).T# NX1
    #print(loss1)
    total_loss = np.sum(np.square(y-prediction)) #scaler ---> #1XN * NX1
    return total_loss #returns scaler
