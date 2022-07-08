Thanks Thanks Thanks Thanks Thanks Thanks import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cost_function import *
from prediction_function import *
import time



filename =  './Prob1_data.csv'
data = pd.read_csv(filename)
print(data.info())
print(data.shape)

print(
    '\033[0;30;45m ---Optimizing Linear Regression Using SGD----\033[0;0m'
  )
print(f"\033[0;32;40mSetting up the Variables \033[0;0m")

#d : number of features
d=10 #bias + number of terms

#Number of iterations
epochs=30
#Numbeer of Samples
N= data.shape[0]
#Stopping Criteria
epsilon = 0.01
#mini_batch_size
batch_size = 1000
#number of iterations
iteration = int(N/batch_size)
#learning rate
eta =(5e-04)/max(100,batch_size)


print(f'The  number fo weight vectors {d}')
print(f'The  number epochs {epochs}')
print(f'The  number of samples {N}')
print(f'The  batch_size {batch_size}')
print(f'The  number of iteration per epoch {iteration}')
print(f'The  learning rate {eta}')
print('\n')

print(f"\033[0;32;40mData Shaping and Preprocessing \033[0;0m")


#Data Shaping and Preprocessing
features = np.asarray(data.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values).T #shape d X N
label = np.asarray(data.label_map.values).reshape(N,-1).T #shap 1XN
print(f'features {features.shape}')
print(features[:,1:3])
print('\n')
print(f'label {label.shape}')
print(label)
print('\n')


#Initialization
print(f"\033[0;32;40mInitializing Weight Vector \033[0;0m")

#wk = np.random.randint(2,100,size=10).reshape(d,1)
wk= np.array([40, 52, 14, 30, 47, 56, 36, 32, 12, 11]).reshape(d,1)
print(f'feature weights (wk) {wk.shape}')

print(f"\033[0;32;40mInitializing Cost Vector \033[0;0m")
J = []
J.append(cost_calc(features,wk,label))

#gradient : first derivative of the cost function
print(f"\033[0;32;40mInitializing Gradient Vector \033[0;0m")
gradient = np.ones(d).reshape(d,1)
print(f'The shape of the gradient vector  {gradient.shape}')

#Optimization
print('\n')
print(f"\033[0;32;40mOptimization in Process \033[0;0m")
#while np.max(abs(gradient))>epsilon:
epoch=0
epoch_time = []

while epoch<epochs:
    epoch_start_time = time.time()
    epoch+=1
    index = np.arange(N)
    #randomizing the Index Essential for SGD
    np.random.shuffle(index)
    #print(index)
    gradient = np.ones(d).reshape(d,1)
    #This jumps to the index of the batch to be picked up depending on the batch size
    for batch in range(0,iteration*batch_size,batch_size):
        current_gradient = np.ones(d).reshape(d,1)
        current_index = index[batch:batch+batch_size]
        current_features = features[:,current_index]
        current_labels   = label[:,current_index]
        current_prediction = pred_funtion(current_features,wk)+ np.random.randn(1, batch_size) #returns 1XN
        gradient += 2*np.dot(current_features,(current_prediction.T-current_labels.T.reshape(batch_size,-1)))  # dXN * (NX1 - NX1) ---> dX1
    wk = wk- eta*(gradient) #(dX1)- scaler*(dX1) ---->dX1
    cost = cost_calc(features,wk,label) #scaler value keep getting added
    J.append(cost)
    #printing functions
    epoch_end_time = time.time()
    epoch_time.append(epoch_end_time - epoch_start_time)
    if epoch % 5 ==0:
        avg_epoch_time = np.sum(epoch_time)/epoch
        print(f'cost after {epoch} epochs {cost:.3f} and average epoch_time is {avg_epoch_time:.2f}')
    if epoch==1:
      print(f'The shape of the mimibatch feature vector {current_features.T.shape}')
      print(f'The shape of the mimibatch label vector {current_labels.T.shape}')
      print(f'The shape of the gradient vector {gradient.shape}')
      print('\n')


print(f"\033[0;32;40m Optimization Done \033[0;0m")
print('\n')
print(f"\033[0;32;40m Final Weights Veector \033[0;0m")
print(wk)


plt.plot(J)
plt.show()
