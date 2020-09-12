import json
import pandas as pd 
import time
import numpy as np
pd.set_option('display.max_colwidth', -1)
##GENERAL
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
import csv
import re
import concurrent.futures
##SPACY
import spacy
nlp = spacy.load('en_core_web_sm')
#nlp = spacy.load('en_core_web_md')
#nlp = spacy.load('en_core_web_lg')
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.pipeline import SentenceSegmenter
from spacy import displacy

#DATA LOADING AND PROFILING
##Reading Through Chunk Size Parameter
data_preprocessing_start_time = time.time() 
chunk_start=time.time()
path = '/Users/ankitkothari/Documents/ONGOING_PROJECTS/optimum/yelp_academic_dataset_review.json'
chunk_iter = pd.read_json(path, lines=True, chunksize=50000)
chunk_list = []
for chunk in chunk_iter:
     chunk_list.append(chunk)
data = pd.concat(chunk_list)
chunk_time = (time.time()-chunk_start)/60
print(f'Time Taken for chunking {chunk_time:03.2f} mins')

print(data.head())
print(f' Memory usage in MB {data.memory_usage(deep=True).sort_values()/(1024*1024)}')
print(f'data types {data.info(memory_usage="deep")}')
print(f'data size {data.size}')

for dtype in ['float','int','object']:
    selected_dtype = data.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

data_reduced = data.copy()
data_int = data_reduced.select_dtypes(include=['int'])
converted_int = data_int.apply(pd.to_numeric, downcast='signed')
print(mem_usage(data_int))
print(mem_usage(converted_int))
compare_ints = pd.concat([data_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['before','after']
print(compare_ints.apply(pd.Series.value_counts))

#Merging new columns
data_reduced[converted_int.columns]=converted_int

#total_time_object=0
total_time_category=0
string_columns=['user_id','business_id']
for col in string_columns:
    num_unique_values = len(data_reduced[col].unique())
    num_total_values = len(data_reduced[col])
    print(f'Ratio of unique values to length of  {col} is {(num_unique_values/num_total_values):03.2f}')
    print(f'Memory Use in in column name {col} Object Data type {col} {data[col].memory_usage(deep=True)/(1024 ** 2):03.2f} MB')
    if num_unique_values / num_total_values < 0.5:
        start_category= time.time()
        data_reduced[col] = data_reduced[col].astype('category')
        total_time_category+=time.time()-start_category
        print(f'Memory use in column name {col} in Category Data type  {col} {data_reduced[col].memory_usage(deep=True)/(1024 ** 2):03.2f} MB')

data_reduced=data_reduced.drop(columns=(['review_id']))
data_preprocessing_optimization_time= (time.time()-data_preprocessing_start_time)/60
print(f'Time Taken for Data Preprocessing Memory Optimization  {data_preprocessing_optimization_time:03.2f} mins')
print(f'row count {data.shape[0]}')
print(f'category time {total_time_category}')
print(data_reduced.head())
print(f' Memory usage in MB {data_reduced.memory_usage(deep=True).sort_values()/(1024*1024)}')
print(f'data types {data_reduced.info(memory_usage="deep")}')
print(f'data size {data_reduced.size}')
print(data_reduced.columns)



#Filtering Time 
filter_start_time= time.time()
only_rating_1 = data_reduced[data_reduced['stars']==1]
print(only_rating_1.shape)
only_rating_1['month'] = only_rating_1['date'].apply(lambda x: x.month)
group_by_rating_1 = only_rating_1.groupby(['business_id','month']).agg(
          {
            'stars': 'count'
          })
optimized_filter = (time.time()-filter_start_time)/60

print(f'Time Taken for Filtering Memory Optimization  {optimized_filter:03.2f} mins')
group_by_rating_1= group_by_rating_1.unstack().reset_index()
print(group_by_rating_1.head())


##Aggregation Time
optimized_aggregation_start_time = time.time()

grouped = data_reduced.groupby(['business_id']).agg(
          {
            'stars':'mean'
          })
grouped = grouped.reset_index()

group_by_rating = grouped = data_reduced.groupby(['stars']).agg(
          {
            'cool': 'sum',
            'funny': 'sum'
          })
group_by_rating = group_by_rating.reset_index()

optimized_aggregation_optimized_time = time.time()-optimized_aggregation_start_time
print(grouped.head(25))
print(group_by_rating.head())
print(f'Time Taken for Aggregarion Memory Optimization  {optimized_aggregation_optimized_time:03.2f} mins')

##Text Cleaning
contraction_mapping = {"ain't": "is not","don't": "do not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have", "w/":"with", "cnt":"cannot", "w/o":"without","u":"you"}

def spacy_preprocessing(text):
    #print(text)
    #text = re.sub(r"\S*\w*.(com)\S*", "",text) #replaces any email or websitw with space
    text = re.sub(r"\b([a-zA-Z]{1})\b", " ", text) #replaces single random characters in the text with space
    text = re.sub(r"[^a-zA-Z]"," ",text) #replaces special characters with spaces
    text = re.sub(r"(.)\1{3,}", r"\1", text) #replaces multiple character with a word with one like pooooost will be post
    text = re.sub(r"\s{2,}", r" ", text) #replaces multiple space in the line with single space
    
    
    tokens = text.split(" ")
    #print(tokens)
    clean_tokens = [contraction_mapping[i] if i in contraction_mapping else i for i in tokens]
    text = " ".join(clean_tokens)
    #except:
    #text=text
    clean_text=[]
    for token in nlp(text):
       if (token.lemma_ != "-PRON-") & (token.text not in nlp.Defaults.stop_words):
           clean_text.append(token.text.lower())
       elif (token.lemma_ == "-PRON-")  & (token.text not in nlp.Defaults.stop_words):
           clean_text.append(token.text.lower())
       else:
           continue
    clean_string = " ".join(clean_text).lstrip()
    #print(type(clean_string))
    return clean_string


checkpoints=[100, 1000,10000,100000]

def text_profiling(checkpoint):
    start = time.time()
    temp= data_reduced.copy()
    temp= temp.iloc[0:checkpoint]
    print(temp.shape)
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    words = pool.map(spacy_preprocessing, temp['text'].to_list())
    words = list(words)
    temp['clean']=words
    end = (time.time()- start)/60
    return temp, end


time_profiling = [text_profiling(checkpoint)[1] for checkpoint in checkpoints]
print(time_profiling)


def text_profile(checkpoint):
    start = time.time()
    temp= data_reduced.copy()
    temp= temp.iloc[0:checkpoint]
    temp['clean']=temp['text'].apply(spacy_preprocessing)
    print(temp.shape)
    end = (time.time()- start)/60
    return temp, end

time_profile = [text_profile(checkpoint)[1] for checkpoint in checkpoints]
print(time_profiling)
print(time_profile)


