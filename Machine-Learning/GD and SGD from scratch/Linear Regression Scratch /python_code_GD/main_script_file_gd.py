import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from cost_function import *
from prediction_function import *
from sklearn import preprocessing
import time


filename =  './Prob1_data.csv'
data = pd.read_csv(filename)
print(data.info())
print(data.shape)

print(
    '\033[0;30;45m ---Optimizing Linear Regression Using SGD----\033[0;0m'
  )
print(f"\033[0;32;40m Setting up the Variables \033[0;0m")

#d : number of features
d=10 #bias + number of terms

#Number of epochs
epochs=50
#Numbeer of Samples
N= data.shape[0]
#learning rate
eta =0.0000001
#Stopping Crepochsia
epsilon = 0.01


features = np.asarray(data.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values).T #shape d X N
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
#wk = np.random.randint(2,100,size=10).reshape(10,1)
wk= np.array([40, 52, 14, 30, 47, 56, 36, 32, 12, 11]).reshape(d,1)
print(f'feature weights (wk) {wk.shape}')

print(f"\033[0;32;40m Initializing Cost Vector \033[0;0m")
J = []
J.append(cost_calc(features,wk,label))

#gradient : first derivative of the cost function
print(f"\033[0;32;40m Initializing Gradient Vector \033[0;0m")
gradient = np.ones(d).reshape(d,1)
print(f'The shape of the gradient vector  {gradient.shape}')

#Optimization
print('\n')
print(f"\033[0;32;40m Optimization in Process \033[0;0m")
#while np.max(abs(gradient))>epsilon:
epoch=0
epoch_time = []
while epoch<epochs:
    epoch_start_time = time.time()
    epoch+=1
    #Step 1: Make Predictions
    new_prediction = pred_funtion(features,wk) #returns 1XN
    #Step2: Calculate the gradient
    gradient = 2*np.dot(features,(new_prediction.T-label.T.reshape(N,-1)))  # dXN * (NX1 - NX1) ---> dX1
    #Step3: Update the weight
    wk = wk- eta*(gradient) #dX1
    #Step4: Calculate the cost 
    cost = cost_calc(features,wk,label) #scaler value keep getting added
    J.append(cost)
    epoch_end_time = time.time()
    epoch_time.append(epoch_end_time - epoch_start_time)
    if epoch % 5 ==0:
        avg_epoch_time = np.sum(epoch_time)/epoch
        print(f'cost after {epoch} epochs {cost:.3f} and average epoch_time is {avg_epoch_time:.2f}')


print(f"\033[0;32;40m Optimization Done \033[0;0m")
print('\n')
print(f"\033[0;32;40m Final Weights Veector \033[0;0m")
print(wk)

plt.plot(J)
plt.show()
