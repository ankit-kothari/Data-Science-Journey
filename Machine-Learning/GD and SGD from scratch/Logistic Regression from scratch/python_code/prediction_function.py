import numpy as np


def pred_funtion(x,wk):
    #print(f'Making Predictions')
    #print('\n')
    #print(f' feature matrix shap {x.shape}')
    #print(f' weigh vector Transpose shape {wk.T.shape}')
    wt_x = wk.T.dot(x) #returns 1XN
    g_new = 1/(1+np.exp(-1*wt_x))
    #print(f' shape of the prediction vector  {g_new.shape}')
    return g_new
