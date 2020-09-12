## Code

[ankit-kothari/optimizing_large_datasets](https://github.com/ankit-kothari/optimizing_large_datasets)

## Introduction

Working with a large dataset is always tricky. Although computing is advancing today at a rapid rate and with the access to GPU's,  things have become much realistic to analyze and model large datasets. Python is one of the most used programming languages in data science with its great libraries and strong community. But when it comes to speed,  Python programs execute as a single process using a single CPU. But most of the computers today have 4 (or more) CPU cores. That means that a lot of computer power is sitting idle while waiting for the program to finish running

In this study, I am working with a yelp data set which is around **6.9GB** ([https://www.yelp.com/dataset](https://www.yelp.com/dataset)). The goal of the article is to apply different techniques of optimization to work and analyze this dataset faster. In this study, I will try to use the following techniques to speed up the process of data loading and analyzing. 

- Data loading with pandas **chunksize** parameter
- Data Types **down-casting** (eg: int64 to int16)
- Using **parallel processing** with pythons **concurrent.futures** library
- Dropping columns which are not needed.

## Evaluation Criteria

I want to evaluate how much these optimization technique affect in the following four categories in terms of time.

- Data Loading and preprocessing
- Data Aggregation
- Data Filtering
- Text Cleaning and profiling at 1000, 10000, 1000000 rows.

## How to achieve most memory improvements?

- We can save memory by converting within the same type (from float64 to float32 for example), or by converting between types (from float64 to int32)
- By converting the string columns to a numeric type.
- Convert all of the columns where the values are **less than 50% unique to the category type**, The category type uses integer values under the hood to represent the values in a column, rather than the raw values. Pandas use a separate mapping dictionary that maps the integer values to the raw ones. This arrangement is useful whenever a column contains a limited set of values. While converting all of the columns to this type is appealing, it's important to be aware of the trade-offs. The biggest one is the inability to perform numerical computations.
- and the string columns that contain numeric values to the float type example percentage with a sign % if we can remove it.
- only columns with not null values can be converted to integers. pandas.to_datetime()
- Pandas provides the datetime64 type, which is a more **memory efficient datatype** to represent dates.
- Identify float columns that contain missing values, and that we can convert to a more space efficient subtype.
- Identify float columns that don't contain any missing values, and that we can convert to the integer type because they represent whole numbers.
- use df = pd.read_csv('data.csv', parse_dates=["StartDate", "EndDate"]), df = pd.read_csv('data.csv', usecols=["StartDate", "EndDate"])

Note that converting from float64 to int64 won't save any memory because values of both types are represented using 8 bytes.

## Concurrent Futures package  for Parallel Processing

- The main advantage of the concurrent.futures package is that it gives you a simple, consistent, interface for both threads and processes.
- Threads are good for situations where:
    - You have long-running I/O bound tasks.
- Threads aren't so good for situations for the following tasks and  Process Pools should work better in those scenarios with **ProcessPoolExecutor class**:
    - You have CPU-bound tasks.
    - You have tasks that will run very quickly (so the overhead outweighs the gains).

## Data Loading and Data Profiling

### Data Loading without optimization

```python
fullfile_start=time.time()
path = '/Users/ankitkothari/Documents/ONGOING_PROJECTS/optimum/yelp_academic_dataset_review.json'
data = pd.read_json(path, lines=True)
full_file_read_time = (time.time()-fullfile_start)/60

**Time Taken for fullfile to read 6.34 mins**
```

### Data Profiling without Optimization

```python
print(data.head())
print(f' Memory usage in MB {data.memory_usage(deep=True).sort_values()/(1024*1024)}')
print(f'data types {data.info(memory_usage="deep")}')
print(f'data size {data.size}')

for dtype in ['float','int','object']:
    selected_dtype = data.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

**Average memory usage for float columns: 0.00 MB
Average memory usage for int columns: 48.96 MB
Average memory usage for object columns: 1358.42 MB**

data_preprocessing_time= (time.time()-data_preprocessing_start_time)/60
print(f'Time Taken for Data Preprocessing WITHOUT Optimization  {data_preprocessing_time:03.2f} mins')

**Time Taken for Data Loading and profiling WITHOUT Optimization  7.37 mins**
```

### Memory Usage by Columns in MB  without optimization

![alt text](https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_6.08.47_PM.png)  

### DataTypes without optimization
![alt text](https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_6.07.28_PM.png)

### Data Loading with optimization

- Using **chunksize** parameter to load the data

```python
chunk_start=time.time()
path = '/Users/ankitkothari/Documents/ONGOING_PROJECTS/optimum/yelp_academic_dataset_review.json'
chunk_iter = pd.read_json(path, lines=True, chunksize=50000)
chunk_list = []
for chunk in chunk_iter:
     chunk_list.append(chunk)
data = pd.concat(chunk_list)
chunk_time = (time.time()-chunk_start)/60
print(f'Time Taken for chunking {chunk_time:03.2f} mins')

**Time Taken for chunking 1.79 mins**
```

### Data Profiling with Optimization

```python
print(f' Memory usage in MB {data.memory_usage(deep=True).sort_values()/(1024*1024)}')
print(f'data types {data.info(memory_usage="deep")}')
print(f'data size {data.size}')

for dtype in ['float','int','object']:
    selected_dtype = data.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

**Average memory usage for float columns: 0.00 MB
Average memory usage for int columns: 48.96 MB
Average memory usage for object columns: 1358.42 MB**

```

- Downcasting the datatypes wherever possible
- The integer columns are evaluated and down-casted to the most appropriate integer format, for example **"stars"** columns has values only between **1-5** so it does not need int64 datatype which ranges from **-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807. This is a waste of memory. We can convert it to int8 which takes in  -128 to 127 which is much more memory efficient.**

```python
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

**Output:**

         **before  after
int8      NaN     1.0  
int16     NaN     3.0  
int64     4.0     NaN**
```

- The **object columns** are evaluated and converted into **Category column** if the ratio of **len(unique values)/len(lenght of the columns) is less than 0.5**

```python
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

Ratio of unique values to length of  user_id is 0.25
Memory Use in column name user_id **Object Data type user_id 604.31 MB**
Memory use in column name user_id in **Category Data type  user_id 258.92 MB**
Ratio of unique values to length of  business_id is 0.03
Memory Use in column name business_id **Object Data type business_id 604.31 MB**
Memory use in column name business_id in **Category Data type  business_id 56.37 MB

Time Taken for Data Preprocessing Memory Optimization  3.52 mins**
```

### Dropping columns which are not used

- Since 'review_id' column consist of unique review id  in object datatype which serves the same purpose of index column in this case, it can be dropped.

```python
data_reduced=data_reduced.drop(columns=(['review_id']))
```

### Memory Usage by Columns **in MB** with optimization
![alt text](https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_7.31.24_PM.png)

### DataTypes with optimization
![alt text](https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_7.31.37_PM.png)

### Optimized Memory Usage vs Original Memory Usage.
<img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-10_at_12.11.44_AM.png","https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_11.19.15_PM.png" width="40%">

### Optimized time for data loading and profiling
<img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_11.19.15_PM.png" width="40%">

## Aggregating

The following test code was run with and without optimization technique. 

```python
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
```

<img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_11.17.48_PM.png" width="40%">

## Filtering

The following test code was run with and without optimization technique. 

```python
only_rating_1 = data_reduced[data_reduced['stars']==1]
print(only_rating_1.shape)
only_rating_1['month'] = only_rating_1['date'].apply(lambda x: x.month)
group_by_rating_1 = only_rating_1.groupby(['business_id','month']).agg(
          {
            'stars': 'count'
          })
optimized_filter = (time.time()-filter_start_time)/60

group_by_rating_1= group_by_rating_1.unstack().reset_index()
```

<img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_11.20.11_PM.png" width="40%">

## Text Cleaning

```python
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
```


<img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_11.29.35_PM.png" width="40%">

- Process Pools use **max_workers** parameter, it depends on the cores the  computer has and can be set accordingly, I had 6 so I have experimented with 4 cores for this process.
- **pool.map**  takes in the function to be applied to each item of the list passed with it, Here **temp['text'].to_list() is a list of reviews, so text cleaning will be done to each of the reviews and returned as one list**

Profiling it with cleaning 1000, 10000, 100000 rows **without Process Pools**                                


```python
checkpoints=[100, 1000,10000,100000]
def text_profile(checkpoint):
    start = time.time()
    temp= data.copy()
    temp= temp.iloc[0:checkpoint]
    temp['clean']=temp['text'].apply(spacy_preprocessing)
    print(temp.shape)
    end = (time.time()- start)/60
    return temp, end

time_profile = [text_profile(checkpoint)[1] for checkpoint in checkpoints]
print(time_profile)

**Time Taken is 27+ mins**
```
Profiling it with cleaning 1000, 10000, 100000 rows **with Process Pools**


```python
checkpoints=[100, 1000,10000,100000]
def text_profiling(checkpoint):
    start = time.time()
    temp= data_reduced.copy()
    temp= temp.iloc[0:checkpoint]
    print(temp.shape)
    **pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
    words = pool.map(spacy_preprocessing, temp['text'].to_list())**
    words = list(words)
    temp['clean']=words
    end = (time.time()- start)/60
    return temp, end
time_profile = [text_profile(checkpoint)[1] for checkpoint in checkpoints]
**Time Taken is 8 mins**
```

## Results

### Time Savings: 69%. The time is reduced from 39 minutes to 12 minutes

<img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_10.15.55_PM.png" width="50%"> <img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_10.02.24_PM.png" width="50%">

### Memory Usage Reduction 23%. The memory usage reduced from 6.9 gb to 5.3 gb
<img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_10.50.21_PM.png" width="50%">

<img src="https://github.com/ankit-kothari/optimizing_large_datasets/blob/master/optimization_images/Screen_Shot_2020-08-09_at_10.58.06_PM.png" width="50%">

 
## Future Work

- Evaluate working with PySpark which works on lazy evaluation and perform time savings calculations.
