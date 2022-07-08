import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from cost_function import *
from prediction_function import *
from sklearn import preprocessing


filename =  './Prob1_data.csv'
data = pd.read_csv(filename)
print(data.info())
print(data.shape)

print(
    '\033[0;30;45m ---Optimizing Linear Regression Using GD Using Normal Equation----\033[0;0m'
  )
print(f"\033[0;32;40m Setting up the Variables \033[0;0m")

#d : number of features
d=10 #bias + number of terms

#Number of iterations
iter=2
#Numbeer of Samples
N= data.shape[0]
#learning rate
eta =0.0000001
#Stopping Criteria
epsilon = 0.01

#Shaping Data
print(f"\033[0;32;40m Scaling Data \033[0;0m")
def normalize(subset,scaler='standard'):
   '''
   subset = dataframe containg all the data
   returns = scaled data for all the int and float colmuns along with the
            non numeric data
   filter:
        scaler type:
        1. MinMax Scaler : 'minmax'
        2. Standard Scaler : 'standard'
        3. Robust Scaler : 'robust'
   '''

   continious_columns = subset.select_dtypes(include=['float']).columns
   if scaler=='minmax':
     mm_scaler = preprocessing.MinMaxScaler()
   elif scaler=='standard':
     mm_scaler = preprocessing.StandardScaler()
   else:
     mm_scaler = preprocessing.RobustScaler()
   subset= mm_scaler.fit_transform(subset[continious_columns])
   subset=pd.DataFrame(subset,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9'])
   return subset, mm_scaler





#Shaping Data
#print(f"\033[0;32;40m Data and Data Dimensions \033[0;0m")
#features,_ = normalize(data.iloc[:,0:9],scaler='standard')
#print(features.shape)
#print(type(features))
#features.loc[:,'x0']=1
#features=features.values.T
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
#wk = np.array([2.5,4,6.8,5,7,]).reshape(d,1)
wk = np.random.randint(2,100,size=10).reshape(10,1)
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
iteration=0
while tqdm(iteration<iter):
    iteration+=1
    new_prediction = pred_funtion(features,wk) #returns 1XN
    #gradient = 2*np.dot(features,(new_prediction.T-label.T.reshape(N,-1)))  # dXN * (NX1 - NX1) ---> dX1
    #wk = wk- eta*(gradient) #dX1


    matrix = np.dot(features,features.T) #makes it dXN dot Nxd ----> dXd
    vector = np.dot(features,label.T)   #dXN -- NX1, ---->d
    print(matrix.shape)
    print(vector.shape)
    wk = np.linalg.solve(matrix,vector)  #---. dX1
    cost = cost_calc(features,wk,label) #scaler value keep getting added
    print(cost)
    J.append(cost)

print(f"\033[0;32;40m Optimization Done \033[0;0m")
print('\n')
print(f"\033[0;32;40m Final Weights Veector \033[0;0m")
print(wk)

plt.plot(J)
plt.show()
