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
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

data = pd.read_csv('./rating.csv', usecols=['userId','movieId','rating','timestamp'])

#Downcasting Data
print(f"\033[0;32;40mOptimization of Memory usage by transforming datatypes\033[0;0m")
print('\n')
print(f"\033[0;32;40mBEFORE\033[0;0m")
print(f'Memory usage in MB \n{data.memory_usage(deep=True).sort_values()/(1024*1024)}')
print(f'data size {data.size}')
print(data.info(memory_usage='deep'))
print('\n')

def mem_usage(pandas_obj):
    '''
    Takes a pandas dataframe and shows it memory usage.
    '''
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return "{:03.2f} MB".format(usage_mb)


#Downcast Integer Data
data_reduced = data.copy()
data_int = data_reduced.select_dtypes(include=['int'])
converted_int = data_int.apply(pd.to_numeric, downcast='signed')

#Downcasting float data
data_float = data_reduced.select_dtypes(include=['float'])
converted_float = data_float.apply(pd.to_numeric, downcast='float')

print('\n')
#comparing before and after downcasting
compare_ints = pd.concat([data_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['before','after']
compare_floats = pd.concat([data_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['before','after']
print(compare_ints.apply(pd.Series.value_counts))
print(compare_floats.apply(pd.Series.value_counts))

print('\n')
print(f"\033[0;32;40mAFTER\033[0;0m")



def make_continious_columns(column):
    """
    Input: Takes in a pandas column which will have ids like 1,4,6,8
    Output: Retruns continious ids like 0,1,2,3
    with a dictionary mapping {1:0,4:1,6:1,8:2}
    """
    unique_values = column.unique()
    movieid_to_movies_idx = {key:idx for idx,key in enumerate(unique_values)}
    return movieid_to_movies_idx , np.array([movieid_to_movies_idx[x] for x in column]), len(unique_values)

def make_continious_dataframe(movies_df):
    """Transform rating data with continuous user and movies ids
    Input dataframe: Takes in a pandas dataframe
    Output it returns 1. movie_ids: Mapping of original movie id to continious column
                      2. num_movies: Total number of movies
                      3. num_users: Total number of users
                      4. user_ids: Mapping of original user ids to continious column
                      5. movie_ids: Mapping of original movie id to continious column
    """

    movies_ids, movies_df.loc[:,'movieId'], num_movies = make_continious_columns(movies_df['movieId'])
    user_ids, movies_df.loc[:,'userId'], num_users = make_continious_columns(movies_df['userId'])
    return movies_df, num_users, num_movies, user_ids, movies_ids


def transform_testing_data(valid_df, user_ids, movies_ids):
    """
    Movie Recommendation has problem with cold start condition
    It only rates movies that it has seen and recommended movies to user id that it has seen.
    This function filters if there is userId or a movieId is in the testing set but now in the training set.
    """
    df_val_chosen = valid_df['movieId'].isin(movies_ids.keys()) & valid_df['movieId'].isin(user_ids.keys())
    valid_df = valid_df[df_val_chosen]
    valid_df.loc[:,'movieId'] =  np.array([movies_ids[x] for x in valid_df['movieId']])
    valid_df.loc[:,'userId'] = np.array([user_ids[x] for x in valid_df['userId']])
    return valid_df


data= pd.read_csv('./rating.csv',usecols=['userId','movieId','rating'], dtype={'userId':'int32','movieId':'int32','rating':'float32'})
print(data.info(memory_usage='deep'))
print('\n')
print(f"\033[0;32;40mPreprocessing MovieIds and UserIds to have continious columns \033[0;0m")



train_df, valid_df = train_test_split(data, test_size=0.4)
valid_df.reset_index()[['userId','movieId','rating']]

new_data, num_users, num_movies, user_ids, movies_ids = make_continious_dataframe(train_df)
new_data_test = transform_testing_data(valid_df, user_ids, movies_ids)
new_data = new_data.reset_index()[['userId','movieId','rating']].rename(columns={'movieId':'movie_idx'})
new_data_test = new_data_test.reset_index()[['userId','movieId','rating']].rename(columns={'movieId':'movie_idx'})

print(f"\033[0;32;40mCreating Training Data and Testing Data\033[0;0m")
#saving the train and test files
new_data.to_csv('./new_data_train.csv')
new_data_test.to_csv('./new_data_test.csv')
print(f"\033[0;32;40mFiles have been created\033[0;0m")
