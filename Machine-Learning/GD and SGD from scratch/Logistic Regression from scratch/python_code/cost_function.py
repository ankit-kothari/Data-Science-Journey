import numpy as np
from prediction_function import *

def cost_calc(x,w,y):
    prediction = pred_funtion(x,w).T # NX1
    loss1 = y.dot(np.log(prediction))  #1XN * NX1 ---> Scaler
    #print(loss1)
    total_loss = loss1 + (1-y).dot(np.log(1-prediction)) #scaler ---> #1XN * NX1
    total_samples = x.shape[1]
    #print(total_samples)
    averge_loss =(-1/total_samples)*total_loss
    return averge_loss #returns scaler
