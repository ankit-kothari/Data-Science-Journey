## Code

[Google Colaboratory](https://colab.research.google.com/drive/1rXDFWIySDIw5a24mNScqU4dLUPsAAsSc?usp=sharing)

## Intended Audience

- Knowledge about basics of Tensorflow 2.0 and Keras
- Knowledge about LSTM, RNN's
- Knowledge about word embeddings

## Dataset Cleaning and Extraction

[NLP PART1: Data Cleaning, Extraction and Topic Modeling](https://www.notion.so/NLP-PART1-Data-Cleaning-Extraction-and-Topic-Modeling-bb571ba8ed4c4014bc7243c5a0d1f233)

- The shape of the initial dataset is (2999999, 3) reviews with columns ['rating',' title', 'review']
- Extracted all reviews with the word or mention of Amazon  to reduce the dataset to  (112106, 3)
- Performed text cleaning.
- Performed Topic Modeling on the dataset using NMF and assigned topics to all the reviews.
- Filtere the data with the following categories  ['books', 'video-quality', 'refund-and-return', 'movies', 'music', 'games']
- The following classification task models the data to predict one of the above categories.

## Word Embeddings

It is used to create a vector relationship between the words in the corpus, There are a number of options,

- Glove, Word2Vec
- Download the pretrained Glove Vector embeddings
- Create a dictionary of word2vector from the corpus in the dataset
- Create an embedding matrix  (we can restric the Max Vocab Size)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e1f11760-c69d-49bd-811c-f1add9436d8e/A3E12406-60DE-49D9-BCB4-77080F0A8724.jpeg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e1f11760-c69d-49bd-811c-f1add9436d8e/A3E12406-60DE-49D9-BCB4-77080F0A8724.jpeg)

## Tokenizer and Padding

- Creating a text to sequecne using TF tokenizer.
- Creating a word2index dictionary.
- Padding to make it a constant sized sequence.

## Model Architechture

### Terminology

```
# N = number of samples
# T = sequence length
# D = number of input features (embedding dimension)
# M = number of hidden units
# K = number of output units
# DU = Dense Units
```

### TF2.0 NLP: Part2 Multi Class Text Classification BiLSTM

[Google Colaboratory](https://colab.research.google.com/drive/1UcBWQVWSDsCKkFNw3qtYNF4qDecWgOYr?usp=sharing)

In this architecture we the sequence once from N(1) to N(T=Sequence length) and then we start from N(T=Sequence length) to N(1). This proves really helpful in remembering long term dependencies.

```python
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))                                              (NxT)
embeddings = embedding_layer(input_)                                                      (NxTXD)
lstm_1 = Bidirectional(LSTM(128, return_sequences=True, return_state=False))(embeddings)  (NxTx2M)
dropout = Dropout(0.3)(lstm_1)
lstm_2 = Bidirectional(LSTM(256, return_sequences=True, return_state=False, dropout=0.3)) (NxTx2M) #2M because of the BiLSTM
lstm_layer = lstm_2(dropout)
gmpl= GlobalMaxPool1D(name='gmpl')(lstm_layer)                                            (Nx2M)
dense = Dense(64,kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
    kernel_regularizer=tf.keras.regularizers.l1(0.01)
    activity_regularizer=tf.keras.regularizers.l2(0.01), activation='relu')(gmpl)         (NxDU)
batch_norm = BatchNormalization()(dense)                                                  (NXK)
dense_1 = Dense(6, activation='softmax')
output = dense_1(batch_norm)
model = Model(input_, output)
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d46d1482-81d7-445f-a8ce-b660014e92e4/Screen_Shot_2020-07-08_at_3.06.24_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d46d1482-81d7-445f-a8ce-b660014e92e4/Screen_Shot_2020-07-08_at_3.06.24_AM.png)

BiLSTM with pre padding Train and Val Loss

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8160df04-6280-40ec-847f-cf03bbf4346c/Screen_Shot_2020-07-08_at_3.06.55_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8160df04-6280-40ec-847f-cf03bbf4346c/Screen_Shot_2020-07-08_at_3.06.55_AM.png)

BiLSTM with pre padding Train and Val accuracy

Best Output: accuracy: 0.9552  ;  val_accuracy: 0.9474

### TF2.0 NLP PART 3: NLP:  Multi Class Text Classification LSTM

[Google Colaboratory](https://colab.research.google.com/drive/1LUSoFn_xlcAODf-WOLb8xLL5p9z0o47a?usp=sharing)

```python
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))                                                           (NxT)
embeddings = embedding_layer(input_)                                                                   (NxTXD)
bilstm = LSTM(32, return_sequences=True, return_state=False, dropout=0.2)(embeddings)                  (NxTXM)
lstm = LSTM(64, return_sequences=True, return_state=False, dropout=0.2)                                (NxTXM)
lstm_layer = lstm(bilstm)
gmpl= GlobalMaxPool1D(name='gmpl')(lstm_layer)                                                         (NXM)
dense = Dense(6)                                                                                       (NXK)
output = dense(gmpl)
model = Model(input_, output)
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/de2f2946-9984-4f7d-8424-c19be51a3a7f/Screen_Shot_2020-07-08_at_2.35.40_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/de2f2946-9984-4f7d-8424-c19be51a3a7f/Screen_Shot_2020-07-08_at_2.35.40_AM.png)

LSTM with pre padding Train and Val Loss

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4787c4e5-81af-4080-9cd4-6bd8a9bae8df/Screen_Shot_2020-07-08_at_2.35.26_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4787c4e5-81af-4080-9cd4-6bd8a9bae8df/Screen_Shot_2020-07-08_at_2.35.26_AM.png)

LSTM with post padding Train and Val accuracy

Best Output:  accuracy: 0.9216  ; val_accuracy: 0.9295

### TF2.0 NLP: Part 4 Multi Class Text Classification BiLSTM with post padding

[Google Colaboratory](https://colab.research.google.com/drive/1J0uNOgxFFtOPofeBbCGeRbkWG1o1SpH6?usp=sharing)

```python
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))                                              (NxT)
embeddings = embedding_layer(input_)                                                      (NxTXD)
lstm_1 = Bidirectional(LSTM(128, return_sequences=True, return_state=False))(embeddings)  (NxTx2M)
dropout = Dropout(0.3)(lstm_1)
lstm_2 = Bidirectional(LSTM(256, return_sequences=True, return_state=False, dropout=0.3)) (NxTx2M) #2M because of the BiLSTM
lstm_layer = lstm_2(dropout)
gmpl= GlobalMaxPool1D(name='gmpl')(lstm_layer)                                            (Nx2M)
dense = Dense(64,kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
    kernel_regularizer=tf.keras.regularizers.l1(0.01)
    activity_regularizer=tf.keras.regularizers.l2(0.01), activation='relu')(gmpl)         (NxDU)
batch_norm = BatchNormalization()(dense)                                                  (NXK)
dense_1 = Dense(6, activation='softmax')
output = dense_1(batch_norm)
model = Model(input_, output)
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21a9c0a4-b0dd-46ec-9851-8dab2e95ee80/Screen_Shot_2020-07-08_at_1.31.21_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21a9c0a4-b0dd-46ec-9851-8dab2e95ee80/Screen_Shot_2020-07-08_at_1.31.21_AM.png)

BiLSTM with post padding Train and Val Loss

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e9a68d76-9164-4235-80cc-76a80998d401/Screen_Shot_2020-07-08_at_1.32.15_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e9a68d76-9164-4235-80cc-76a80998d401/Screen_Shot_2020-07-08_at_1.32.15_AM.png)

BiLSTM with post padding Train and Val accuracy

Best Output:  accuracy: 0.9099 ; val_accuracy: 0.9090

### TF2.0 NLP: Part5 Multi Class Text Classification CNN-1D

[Google Colaboratory](https://colab.research.google.com/drive/1by90KYnHm2h77lJrVWiUS3AogsV9kqBF?usp=sharing)

```python
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
embeddings = embedding_layer(input_)
drop_embed_layer = SpatialDropout1D(.2, name='drop_embed')(embeddings)

conv1 = Conv1D(256, 20,strides=1, activation='relu')(drop_embed_layer)
maxp_1 = GlobalMaxPool1D(name='maxp_1')(conv1)

conv2= Conv1D(256, 10, activation='relu' )(drop_embed_layer)
maxp_2 = GlobalMaxPool1D(name='maxp_2')(conv2)

conv3= Conv1D(256, 5, activation='relu' )(drop_embed_layer)
maxp_3 = GlobalMaxPool1D(name='maxp_3')(conv3)

concat = concatenate([maxp_1, maxp_2, maxp_3])

dense = Dense(64,kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
    kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.05), activation='relu')(concat)
batch_norm = tf.keras.layers.Dropout(0.2)(dense)

output = Dense((len(K))(batch_norm)
model = Model(input_, output)
```

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b7833787-a327-4ad4-8b95-b10205295d3e/Screen_Shot_2020-07-08_at_3.46.30_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b7833787-a327-4ad4-8b95-b10205295d3e/Screen_Shot_2020-07-08_at_3.46.30_AM.png)

Multi-Layer CNN-1D with pre-padding Train and Val Loss

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/273988b6-7ce0-4df0-8d91-ad0b3fabc66e/Screen_Shot_2020-07-08_at_3.46.20_AM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/273988b6-7ce0-4df0-8d91-ad0b3fabc66e/Screen_Shot_2020-07-08_at_3.46.20_AM.png)

Multi-Layer CNN-1D with pre-padding Train and Val accuracy

Output: accuracy: 0.9078  ;  val_accuracy: 0.9144

## Model Outputs (epochs:10)

[Model outputs by Architeture ](https://www.notion.so/999de183ae1942dc9f9d32f35bbdff56)
