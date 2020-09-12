# Topic-Modeling
[Google Colaboratory](https://colab.research.google.com/drive/1idYU7lPIgxO87KQLtFcCNJe1ALZWLlWq?usp=sharing)

Link to the google-colab workbook for Topic Modeling

## Data Cleaning

- Replaces any email or website with space
- Replaces single random characters in the text with space
- Replaces special characters with spaces
- Replaces multiple character with a word with one  e.g. like pooooost will be post
- Replaces multiple space in the line with single space
- Remove all the stop words
- Perform Lemmatization of words.
- Lowercasing the words
- Removing all the leading spaces in each of the reviews
- Dropping all the rows with no Reviews

[Comparison  Original Text vs Clean Text]

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/tp2.png" width="90%">

## **Comparison between Spark and Pandas Transformation**

### Spark Transformation

- Reading the csv file into spark DataFrame.
- Creating a custom function using UDF
- Creating a new column which is the cleaned version of Reviews column using spacy and regex.
- Convert it to pandas for LDA and NMF topic modeling

### Pandas Transformation

- Reading the csv file using the pandas read_csv function.
- Creating a new column which is the cleaned version of Reviews column using spacy and regex.

### Result

- This transformation is done on a CPU
- The dataset used to compare here consist of 24714 rows and one column.
- The Spark Transformation took 332 seconds vs 384 seconds for pandas transformation to achieve the same thing
- So the savings of 52 seconds can be huge and can aggregate if the dataset is huge and frequent analysis is done.

## LDA Implementation

- LDA  is an unsupervised technique so there is no labeling of topics that the model will be trained on.
- Latent topics can be found by searching for group of words that frequently occur together in documents across the corpus. What LDA does is that when you fit it with all those "Reviews" in this case, it is trying its best to find the best topic mix and the best word mix.
- We  must decide on the amount of topics present in the document.
- We also have to interpret the topic using the word distribution
- Process of LDA model
    - Go through each document, and randomly assign each word in the document to one of the N topics.
    - This random assignment already gives  initial topic representations of all the documents and word distributions of all the topics
    - p(topic t | document d) = the proportion of words in document d that are currently assigned to topic t.
    - p(word w | topic t) = the proportion of assignments to topic t over all documents that come from this word w.
    - Reassign w a new topic, where we choose topic t with probability p(topic t | document d) * p(word w | topic t) (probablity that topic t generated word w)
    - After repeating these steps multiple times we reach a steady state where the assignments are acceptable

Documents are probablity distribution over latent topics 

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-06-23_at_1.51.16_AM.png" width="50%">


Topics are themselves probablity distribution over words (Vocublary of words in the whole dataset)


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-06-23_at_1.51.42_AM.png" width="50%">

### Topics:

```python
{0: ['colors', 'don', 'warm', 'quality', 'little', 've', 'cute', 'got', 'time', 'dry', 'sale', 'color', 'size', 'bought', 'buttons', 'price', 'washed', 'wash', 'wear', 'love', 'like', 'coat', 'fabric', 'soft', 'sweater'],

 1: ['usually', 'retailer', 'bought', 'bit', 'sleeves', 'online', 'regular', 'sweater', 'little', 'look', 'tried', 'medium', 'wear', 'long', 'store', 'length', 'love', 'xs', 'petite', 'fit', 'small', 'ordered', 'color', 'like', 'size'],

 2: ['arms', 'cut', 'ordered', 'hips', 'big', 'cute', 'chest', 'short', 'nice', 'bit', 'jeans', 'tight', 'great', 'material', 'little', 'love', 'large', 'shirt', 'waist', 'look', 'fabric', 'small', 'like', 'size', 'fit'], 

3: ['casual', 'little', 'sweater', 'nice', 'looks', 'true', 'colors', 'got', 'dress', 'flattering', 'shirt', 'fall', 'cute', 'bought', 'summer', 'fit', 'fits', 'soft', 'color', 'jeans', 'size', 'perfect', 'wear', 'great', 'love'], 

4: ['great', 'sale', 'slip', 'store', 'looks', 'small', 'tried', 'think', 'look', 'online', 'dresses', 'petite', 'fits', 'perfect', 'quality', 'price', 'love', 'ordered', 'wear', 'fabric', 'beautiful', 'fit', 'size', 'like', 'dress'], 

5: ['waist', 'work', 'pattern', 'soft', 'print', 'length', 'design', 'makes', 'material', 'looks', 'little', 'beautiful', 'nice', 'fit', 'great', 'cut', 'look', 'wear', 'colors', 'like', 'fabric', 'flattering', 'skirt', 'love', 'dress']}
```

### Mapping

```python
mapping ={'0':'Sweater/Jackets','1':'Dresses','2':'Bottom/Jeans','3':'Shirt/Tops','4':'Dresses','5':'Bottom/Jeans'}
```

## NMF Implementation

- NMF is an unsupervised learning algorithm that simultaneously perform dimensionality reduction and clsutering.
- It is used with TF-IDF to model topics across documents.
- Using the original matrix (A), NMF will give you two matrices (W and H). W is the topics it found and H is the coefficients (weights) for those topics. In other words, A is articles by words (original), H is articles by topics and W is topics by words.


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-06-23_at_3.00.43_AM.png" width="50%">


### Topics

```python
THE TOP 15 WORDS FOR TOPIC #0
[['run', 0.4412], ['lbs', 0.4975], ['fits', 0.5175], ['big', 0.5704], ['xs', 0.6369], ['true', 0.6424], ['fit', 0.7021], ['wear', 0.7119], ['usually', 0.805], ['runs', 0.8942], ['medium', 1.065], ['ordered', 1.0686], ['large', 1.1593], ['small', 1.9073], ['size', 2.5544]]

THE TOP 15 WORDS FOR TOPIC #1
[['lovely', 0.1206], ['gorgeous', 0.1211], ['knee', 0.1229], ['fabric', 0.1241], ['belt', 0.1254], ['wedding', 0.1504], ['easy', 0.1734], ['dresses', 0.2116], ['summer', 0.2151], ['wear', 0.2298], ['slip', 0.2434], ['perfect', 0.2544], ['flattering', 0.3015], ['beautiful', 0.3465], ['dress', 3.6979]]

THE TOP 15 WORDS FOR TOPIC #2
[['style', 0.1894], ['super', 0.2026], ['got', 0.2299], ['beautiful', 0.2982], ['absolutely', 0.3331], ['wear', 0.3504], ['flattering', 0.371], ['bought', 0.4036], ['fits', 0.432], ['colors', 0.4702], ['perfect', 0.5089], ['color', 0.6037], ['soft', 0.6757], ['sweater', 0.8306], ['love', 2.8862]]

THE TOP 15 WORDS FOR TOPIC #3
[['little', 0.4121], ['cut', 0.4126], ['bit', 0.4217], ['skirt', 0.4277], ['waist', 0.4464], ['short', 0.4472], ['didn', 0.4608], ['color', 0.4767], ['material', 0.4916], ['good', 0.4974], ['fit', 0.5008], ['nice', 0.6976], ['look', 0.7769], ['fabric', 0.871], ['like', 1.3458]]

THE TOP 15 WORDS FOR TOPIC #4
[['looks', 0.1042], ['got', 0.1126], ['tee', 0.1249], ['shirts', 0.1299], ['flattering', 0.1345], ['underneath', 0.1492], ['wear', 0.1616], ['material', 0.1646], ['boxy', 0.1669], ['soft', 0.1734], ['little', 0.2267], ['white', 0.236], ['super', 0.2977], ['cute', 0.9465], ['shirt', 3.0829]]

THE TOP 15 WORDS FOR TOPIC #5
[['boots', 0.2747], ['black', 0.2923], ['quality', 0.3415], ['fall', 0.3774], ['pair', 0.3876], ['leggings', 0.4066], ['summer', 0.4087], ['skinny', 0.4428], ['perfect', 0.4893], ['wear', 0.5173], ['looks', 0.5984], ['fit', 0.7081], ['pants', 0.7267], ['jeans', 1.4255], ['great', 3.0989]]
```

### Mapping

```python
mapping ={'0':'Tops','1':'Dresses','2':'Dresses','3':'Sweater/Jackets','4':'Shirts','5':'Bottom/Jeans'}
```

### Word Cloud

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/tp1.png" width="100%">
