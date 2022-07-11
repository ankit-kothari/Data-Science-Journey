**There are three notebooks attached in this Repo**

1. without_stopwords_deep_learning.ipynb
2. with_stopwords_deep_learning.ipynb
3. fine_tuning_imdb_deep_learning.ipynb (Fine Tuning)
4. custom_model_imdb_deep_learning.ipynb

## Dataset

- Used Huggingface `load_dataset` to get the imdb dataset
    - 25000 training samples
    - 25000 test samples

## Splitting  in Train and Test and Preprocessing the Data

- The data is split into train and test
- Keras tokenizer is used to count the frequency of words (top 5000) on the training data only
    - `with stopwords`
    - `without stopwords`
- Then filtered the reviews and kept only the top 5000 words found in step2  and also removed white spaces, trailing spces and lowercased
- Used Dask to do parallel processing.

## Creating Embedding

- Used the DistilBERT tokenizer to tokenize the data
- Use the DistilBERT model to get the embeddings of the last hidden layer and for the [CLS] token
- The dimension came out `25000,768` for both train and test data

## Comparison of Different Modelss

- It can be seen that general model performacne has reduced by removing the stopwords except for ANN whose performace increased by `5%` after removing the stopwords.
- `DistilBERT Embedding` + Logistic Regression gave the 2nd best result of 85%
- The best results were given by `DistilBERT fine tuning` of around 89%

| Model  | Stopwords used | Accuracy |
| --- | --- | --- |
| DistilBERT Embedding + Logistic Regression | YES | 85% |
| DistilBERT Embedding + Logistic Regression | NO | 84% |
| DistilBERT Embedding + ANN (Feed Forward Network) | YES | 77% |
| DistilBERT Embedding + ANN (Feed Forward Network) | NO | 82% |
| DistilBERT Embedding + CNN Model 1 (Concatenation) | YES | 79% |
| DistilBERT Embedding + CNN Model 1 (Concatenation) | NO | 77% |
| DistilBERT Embedding +CNN Model 2 (Sequential) | YES | 83% |
| DistilBERT Embedding +CNN Model 2 (Sequential) | NO | 82% |
| DistilBERT Embedding Fine Tuning | YES | 89.80 % |
| Custom Model with DistilBERT Embedding Fine Tuning | YES | 90.0 % |