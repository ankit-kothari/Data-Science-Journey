import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from copy import deepcopy
from predict_recommendation import *
from compute_loss import *
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import warnings
warnings.filterwarnings('ignore')


#Importing Train and Test Data
new_data= pd.read_csv('./new_data_train.csv',dtype={'userId':'int32','movie_idx':'int32','rating':'float32'})
new_data_test= pd.read_csv('./new_data_test.csv',dtype={'userId':'int32','movie_idx':'int32','rating':'float32'})


#Number of Users
N=new_data.userId.nunique()
#Movies in train and test set
M = new_data.movie_idx.nunique()
print("N:", N, "M:", M)
#Latent Dimension after Matrix Factorization
K=15
user_matrix = 12*np.random.random((N, K))/K
user = 10*np.ones(N)
movie_matrix = 12*np.random.random((M, K))/K
movie = 10*np.ones(M)
global_mean = new_data['rating'].mean()


print(f'Shape of original Metrix {N}x{M}')
print(f'Shape of user_matrix Metrix {N}x{K}')
print(f'Shape of movie_matrix Metrix {M}x{K}')
print(f'The shrink in paramters is from {N*M} to {(N*K)+(K*M)} due to MF i.e is {100*(((N*K)+(K*M))/(N*M)):.2f}% of the original numbers')
print(f'N {N}')
print(f'M {M}')
print(f'user_matrix shape {user_matrix.shape}')
print(f'movie_matrix shape {movie_matrix.shape}')
print(f'user_bias shape {user.shape}')
print(f'movie_bias shape {movie.shape}')
print(f'Unique Users  {new_data.userId.nunique()}')
print(f'Max, Min UserId,  {new_data.userId.max()},{new_data.userId.min()}')
print(f'Unique Movies {new_data.movie_idx.nunique()}')
print(f'Max, Min MovieId,  {new_data.movie_idx.max()},{new_data.movie_idx.min()}')


#Training Parameters
epochs=[]
train_losses = []
test_losses = []
epochs = 110
reg =.25 # regularization penalty
eta = 0.01

#Initialization Cost Parameters
print(f"\033[0;32;40mInitializing Cost Vector \033[0;0m")
J = []
J.append(cost(new_data,user_matrix,movie_matrix,user,movie,global_mean))
J_test=[]
J_test.append(cost(new_data,user_matrix,movie_matrix,user,movie,global_mean))

#gradient : first derivative of the cost function
print(f"\033[0;32;40mInitializing Gradient Vector \033[0;0m")
user_matrix_gradient = np.zeros((N, K))
movie_matrix_gradient = np.zeros((M, K))
user_gradient = np.zeros(N).reshape(N,1)
movie_gradient = np.zeros(M).reshape(M,1)
print(f'user_matrix_gradient {user_matrix_gradient.shape}')
print(f'movie_matrix_gradient {movie_matrix_gradient.shape}')
print(f'user_gradient {user_gradient.shape}')
print(f'movie_gradient {movie_gradient.shape}')
print(f'Initial Train Cost {J[0]:.2f}')
print(f'Initial Test Cost {J_test[0]:.2f}')

#Functions to calculate user and movie bias
def movie_b2(x):
    '''
    Input: Applies this function to groupby UserId,
    Output: Average rating by UserId across all movies it has rated
    '''
    x['c_q'] = x['c_q'].sum()/(x['movie_idx'].count())
    return x

def user_b(x):
    '''
    Input: Applies this function to groupby movie_idx,
    Output: Average rating for a movie_idx  across all the users that rated this movie.
    '''
    x['b_q'] = x['b_q'].sum()/(x['userId'].count())
    return x


#Optimization
print('\n')
print(f"\033[0;32;40mOptimization in Process \033[0;0m")
#Calculating the actual ratings and storing it in the Compressed Sparse Row Matrix Format
actual_rating = transform2_sparse_matrix(new_data, user_matrix.shape[0], movie_matrix.shape[0],column_name="rating")

epoch=0
epoch_time = []
while epoch<epochs:
    epoch_start_time = time.time()
    epoch+=1

    #Step 1: Make Predicts
    new_prediction = transform2_sparse_matrix(predict(new_data, user_matrix, movie_matrix,user,movie,global_mean), user_matrix.shape[0], movie_matrix.shape[0], 'prediction')


    #Step 2: Update the user_matrix
    new_data['user_gradient']=(-2/new_data.shape[0])*(new_data['rating']-np.sum(np.multiply(user_matrix[new_data['userId']],movie_matrix[new_data['movie_idx']]), axis=1,dtype=np.float32)-user[new_data['userId']]-movie[new_data['movie_idx']]-global_mean)
    user_matrix_gradient = transform2_sparse_matrix(new_data,user_matrix.shape[0], movie_matrix.shape[0],'user_gradient')*movie_matrix +2*reg*user_matrix
    user_matrix = user_matrix - eta*user_matrix_gradient

    #Step 3: Update the user bias matrix
    b_q = -2*(new_data['rating']-np.sum(np.multiply(user_matrix[new_data['userId']],movie_matrix[new_data['movie_idx']]), axis=1)-movie[new_data['movie_idx']]-global_mean) +2*reg*user[new_data['userId']]
    new_data['b_q']=b_q
    new_data = new_data.groupby('userId').apply(user_b)
    user=(user-eta*(new_data.groupby(['userId'])['b_q'].mean().values).reshape(N))

    #Step 4: Update the movie matrix
    new_data['movie_gradient']=(-2/new_data.shape[0])*(new_data['rating']-np.sum(np.multiply(user_matrix[new_data['userId']],movie_matrix[new_data['movie_idx']]), axis=1)-user[new_data['userId']]-movie[new_data['movie_idx']]-global_mean)
    movie_matrix_gradient = transform2_sparse_matrix(new_data,user_matrix.shape[0], movie_matrix.shape[0],'movie_gradient').T*user_matrix+2*reg*movie_matrix
    movie_matrix = movie_matrix - eta*movie_matrix_gradient


    #Step 5: Update the movie bias matrix
    c_q = -2*(new_data['rating']-np.sum(np.multiply(user_matrix[new_data['userId']],user_matrix[new_data['userId']]), axis=1)-user[new_data['userId']]-global_mean)+2*reg*movie[new_data['movie_idx']]
    new_data['c_q']=c_q
    new_data = new_data.groupby('movie_idx').apply(movie_b2)
    movie=(movie-eta*new_data.groupby(['movie_idx'])['c_q'].mean().values).reshape(M)


    #Step6: Calculate the Train and Test cost
    loss = cost(new_data,user_matrix,movie_matrix,user,movie,global_mean) #scaler value keep getting added
    loss_test = cost(new_data_test,user_matrix,movie_matrix,user,movie,global_mean)
    J.append(loss)
    J_test.append(loss_test)
    epoch_end_time = time.time()
    epoch_time.append(epoch_end_time - epoch_start_time)
    if epoch % 5 ==0:
        avg_epoch_time = epoch_time[epoch-1]#np.sum(epoch_time)/epoch
        print(f'Train cost after {epoch} epochs {loss:.3f} and Test cost {loss_test:.3f}  and average epoch_time is {avg_epoch_time:.2f}')


#Plotting thee Training and Test cost
plot_train_loss = go.Figure().add_scatter(y=J).update_layout(title='Training Loss',yaxis=dict(
        title="Mean Squared Error",
        titlefont=dict(
            color="#151515"
        ),
        tickfont=dict(
            color="#151515"
        )))
plot_test_loss = go.Figure().add_scatter(y=J_test).update_layout(title='Test Loss',yaxis=dict(
        title="Mean Squared Error",
        titlefont=dict(
            color="#151515"
        ),
        tickfont=dict(
            color="#151515"
        )))

plot_train_loss.show()
plot_test_loss.show()

#Analysis on the prediction
print(new_data_test.sample(25).sort_values(by='rating'))
print(new_data.groupby(['rating'])['prediction'].mean())
print(new_data_test.groupby(['rating'])['prediction'].mean())
