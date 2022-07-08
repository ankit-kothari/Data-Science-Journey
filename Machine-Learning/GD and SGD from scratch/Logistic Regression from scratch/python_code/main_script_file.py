import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from cost_function import *
from prediction_function import *


filename =  './myData.csv'
data = pd.read_csv(filename)
print(data)

print(
    '\033[0;30;45m ---Optimizing Logistic Regression Using GD----\033[0;0m'
  )
print(f"\033[0;32;40m Setting up the Variables \033[0;0m")

#d : number of features
d=3 #bias + number of terms

#Number of iterations
iter=12
#Numbeer of Samples
N= data.shape[0]
#learning rate
eta =0.05;
#Stopping Criteria
epsilon = 0.01

#Shaping Data
print(f"\033[0;32;40m Data and Data Dimensions \033[0;0m")

features = np.asarray(data.iloc[:,[1,2,3]].values).T #shape d X N
label = np.asarray(data.label_map.values).reshape(N,-1).T #shap 1XN
print(type(label))
print(f'features {features.shape}')
print(features[:,1:3])
print('\n')
print(f'label {label.shape}')
print(label)
print('\n')


#Initialization
print(f"\033[0;32;40m Initializing Weight Vector \033[0;0m")
wk = np.array([2.5,4,6.8]).reshape(d,1)
print(f'feature weights (wk) {wk.shape}')

print(f"\033[0;32;40m Initializing Cost Vector \033[0;0m")
J = []
J.append(cost_calc(features,wk,label)[0][0])

#gradient : first derivative of the cost function
print(f"\033[0;32;40m Initializing Gradient Vector \033[0;0m")
gradient = np.ones(d).reshape(d,1)
print(f'The shape of the gradient vector  {gradient.shape}')

#Optimization
print('\n')
print(f"\033[0;32;40m Optimization in Process \033[0;0m")
while np.max(abs(gradient))>epsilon:
    new_prediction = pred_funtion(features,wk) #returns 1XN
    gradient = (1/N)*np.matmul(features,(new_prediction.T-label.T.reshape(N,-1)))  # dXN * (NX1 - NX1) ---> dX1
    wk = wk- eta*(gradient) #dX1
    cost = cost_calc(features,wk,label) #scaler value keep getting added
    J.append(cost[0][0])

print(f"\033[0;32;40m Optimization Done \033[0;0m")
print('\n')
print(f"\033[0;32;40m Final Weights Veector \033[0;0m")
print(wk)

plt.plot(J)
plt.show()
