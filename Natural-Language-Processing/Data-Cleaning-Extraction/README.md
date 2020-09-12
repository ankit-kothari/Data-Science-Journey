## Code

[Google Colaboratory](https://colab.research.google.com/drive/1rXDFWIySDIw5a24mNScqU4dLUPsAAsSc?usp=sharing)

## Data Preprocessing

### Data Extraction

- Aggregating the ratings

```python
def frametransform(rating):
  if rating == 5 or rating == 4:
    return "Good"
  elif rating == 3:
    return "Average"
  else:
    return "Bad"
```

- Extracting all reviews from amazon and it can be in the form of the following in the review column

```python
pattern =r'\b([Aa]mazon[.+(a-z)]*)'
data= data[data['review'].str.contains(pattern, na=False)]data.head()
```

```
 'Amazon', 'Amazon.', 'Amazon.com', 'amazon', 'amazon.com',
       'amazon.comi', 'Amazon...ever.', 'amazon.(note', 'amazon.)',
       'Amazon)', 'Amazona', 'amazon.com.', 'amazon...read',
       'Amazon(which', 'amazonian', 'Amazon.com(through', 'Amazon.com.',
       'Amazon..', 'amazon.', 'amazon.only', 'Amazon...',
       'Amazon....nothing', 'Amazonians', 'amazon).', 'Amazon.com...',
       'amazon...', 'Amazon).', 'amazon.com)', 'Amazon...couldn',
       'amazon.odder', 'Amazon.ca', 'Amazon..but', 'Amazon.com.uk',
       'Amazon.co.uk', 'Amazon.com)....AND MANY MORE
```

### Data Cleaning

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

### Data Manipulation

- Replace all the extracted patters containing "Amazon" with Amazon so as to make one standard reference everywhere.

```python
data['review']=data['review'].str.replace(pattern, "Amazon")
```

## Topic Modeling

### Identified Topics in the Reviews

I have used the NMF technique to categorize the topics and then drill down into the bad reviews and identify the real issue in the Amazon products/procss.

```python
THE TOP 15 WORDS FOR TOPIC #0
[['battery', 0.9314], ['got', 0.934], ['reviews', 0.9593], ['ve', 0.9597], ['return', 0.9675], ['didn', 0.9704], ['new', 0.9708], ['unit', 1.0729], ['time', 1.1914], ['buy', 1.3241], ['like', 1.3622], ['don', 1.4351], ['use', 1.4604], ['work', 1.5361], ['bought', 1.9406]]

THE TOP 15 WORDS FOR TOPIC #1
[['cover', 0.1862], ['page', 0.1935], ['information', 0.2009], ['copy', 0.2234], ['review', 0.2241], ['com', 0.2377], ['written', 0.2486], ['reviews', 0.2585], ['story', 0.2717], ['reading', 0.352], ['author', 0.3824], ['pages', 0.3916], ['books', 0.5073], ['read', 0.8374], ['book', 4.6406]]

THE TOP 15 WORDS FOR TOPIC #2
[['ray', 0.1929], ['review', 0.1944], ['copy', 0.2051], ['like', 0.213], ['video', 0.2182], ['acting', 0.219], ['love', 0.2311], ['saw', 0.2424], ['seen', 0.2943], ['watching', 0.2943], ['watched', 0.3011], ['story', 0.3061], ['film', 0.3941], ['movies', 0.6997], ['movie', 4.7848]]

THE TOP 15 WORDS FOR TOPIC #3
[['listening', 0.2336], ['cds', 0.2516], ['com', 0.2586], ['sound', 0.2598], ['love', 0.2872], ['like', 0.3178], ['heard', 0.3265], ['track', 0.3343], ['tracks', 0.3681], ['listen', 0.3744], ['song', 0.6414], ['songs', 0.9286], ['album', 0.9673], ['music', 1.0887], ['cd', 3.6868]]

THE TOP 15 WORDS FOR TOPIC #4
[['quality', 0.241], ['vhs', 0.261], ['video', 0.2822], ['blu', 0.2909], ['copy', 0.2919], ['film', 0.2926], ['players', 0.2978], ['ray', 0.3188], ['version', 0.3262], ['dvds', 0.3549], ['disc', 0.3879], ['region', 0.493], ['play', 0.6348], ['player', 0.8039], ['dvd', 4.3705]]

THE TOP 15 WORDS FOR TOPIC #5
[['shipped', 0.4354], ['weeks', 0.4375], ['arrived', 0.486], ['shipping', 0.6109], ['company', 0.6153], ['time', 0.6225], ['refund', 0.6392], ['days', 0.6636], ['customer', 0.7482], ['sent', 0.9724], ['seller', 0.9989], ['service', 1.0031], ['received', 1.9441], ['ordered', 2.0198], ['order', 2.2725]]

THE TOP 15 WORDS FOR TOPIC #6
[['buy', 0.1038], ['information', 0.105], ['use', 0.1065], ['return', 0.1091], ['packaging', 0.1161], ['company', 0.1314], ['disappointed', 0.1324], ['recommend', 0.1354], ['com', 0.1839], ['review', 0.1965], ['purchase', 0.2009], ['purchased', 0.201], ['products', 0.2301], ['description', 0.3262], ['product', 3.9061]]

THE TOP 15 WORDS FOR TOPIC #7
[['love', 0.2529], ['better', 0.2682], ['pay', 0.2756], ['cheaper', 0.295], ['paid', 0.3166], ['local', 0.3169], ['purchase', 0.3352], ['worth', 0.3433], ['stores', 0.3518], ['buy', 0.4311], ['store', 0.4526], ['free', 0.5252], ['best', 0.7121], ['shipping', 0.9957], ['price', 3.3105]]

THE TOP 15 WORDS FOR TOPIC #8
[['disappointed', 0.1321], ['picture', 0.1484], ['wrong', 0.164], ['received', 0.1741], ['seller', 0.1854], ['shipped', 0.1867], ['refund', 0.1952], ['purchase', 0.214], ['items', 0.2208], ['shipping', 0.2259], ['description', 0.2332], ['returned', 0.2792], ['purchased', 0.2956], ['return', 0.528], ['item', 3.724]]

THE TOP 15 WORDS FOR TOPIC #9
[['like', 0.1519], ['nt', 0.1541], ['version', 0.1639], ['son', 0.1666], ['old', 0.1691], ['computer', 0.1717], ['video', 0.1831], ['card', 0.1866], ['graphics', 0.2455], ['playing', 0.2669], ['played', 0.3253], ['fun', 0.4414], ['games', 0.6463], ['play', 0.7434], ['game', 3.9705]]

THE TOP 15 WORDS FOR TOPIC #10
[['series', 0.1805], ['paperback', 0.1825], ['text', 0.1866], ['reader', 0.1923], ['buy', 0.2268], ['love', 0.2398], ['device', 0.2522], ['download', 0.2531], ['reading', 0.3243], ['available', 0.3272], ['read', 0.5825], ['edition', 0.6666], ['version', 1.3391], ['books', 1.478], ['kindle', 2.9971]]

THE TOP 15 WORDS FOR TOPIC #11
[['looks', 0.1377], ['watching', 0.1401], ['prime', 0.1446], ['wrist', 0.1579], ['tv', 0.1653], ['episodes', 0.1862], ['episode', 0.2105], ['love', 0.2211], ['video', 0.2683], ['series', 0.2741], ['watches', 0.3088], ['band', 0.3502], ['time', 0.3935], ['season', 0.4277], ['watch', 3.9986]]

THE TOP 15 WORDS FOR TOPIC #12
[['pieces', 0.3392], ['damaged', 0.3469], ['came', 0.3522], ['episodes', 0.3664], ['missing', 0.3791], ['discs', 0.4312], ['series', 0.4361], ['opened', 0.4381], ['packaging', 0.4853], ['arrived', 0.5352], ['broken', 0.6267], ['disc', 0.7427], ['season', 0.7826], ['box', 2.8086], ['set', 3.717]]

THE TOP 15 WORDS FOR TOPIC #13
[['quickly', 0.1499], ['job', 0.1611], ['recommend', 0.1612], ['happy', 0.1677], ['condition', 0.1782], ['got', 0.1821], ['thank', 0.1911], ['deal', 0.226], ['fast', 0.2325], ['easy', 0.2783], ['works', 0.295], ['service', 0.372], ['thanks', 0.4072], ['love', 0.4673], ['great', 3.7637]]

THE TOP 15 WORDS FOR TOPIC #14
[['overall', 0.1573], ['little', 0.1695], ['album', 0.1988], ['looks', 0.2047], ['video', 0.214], ['condition', 0.2394], ['picture', 0.2637], ['poor', 0.2667], ['bad', 0.2697], ['pretty', 0.2766], ['better', 0.3359], ['sound', 0.5521], ['like', 0.586], ['quality', 1.345], ['good', 3.3109]]
```

### Topic Mapping

```python
topic_mapping={0:'refund-and-return', 1:'books',2:'movies',3:'music',4:'video-quality',5:'shipping',6:'refund-and-return',7:'cost',8:'refund-and-return',9:'games', 10:'books', 11:'tv-series',12:'packaging',13:'deals',14:'product-quality'}
clean_data['topic_name']=clean_data['topic_category'].map(topic_mapping) 
```

### Grouping the data

```python
clean_data_grouped = clean_data.groupby(['topic_name','review_category']).agg(
  {

    'review_category':'count'

  })

```

### Identifying the Topic with highest % of Bad reviews

```python
clean_data_grouped.sort_values(by=['percentage_bad'], ascending=False)
```
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-06_at_10.57.38_PM.png" width="40%">
