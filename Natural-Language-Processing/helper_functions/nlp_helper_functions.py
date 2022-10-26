import json
import re
import sys
import gzip
import codecs
import string
from math import log2
from collections import Counter
import spacy
from spacy.lang.en import English
from traceback_with_variables import activate_by_import
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
# Import functions from other assignments
# 1. Load stopword
# 2. normalize_tokens
# 3. ngrams
# 4. filter_punctuation_bigrams
# 5. filter_stopword_bigrams
# 6. spacy_preprocessing (Text Cleaning)
# 7. word to emotion mapping
# 8. word to polarity Mapping
# 9. word to emotion matrix
# 10. get verb - dependency tag pairs and counts
# 11. Scaling Data
# 12. split_training_set
# 13. plot_confusion_matrix










def load_stopwords(filename):
    stopwords = [] # ASSIGNMENT: replace this with your code
    with open(filename, "r") as f:
      stopwords = [] # ASSIGNMENT: replace this with your code
      for line in tqdm(f):
        line = re.sub(r"\n","",line, flags=re.I)
        stopwords.append(line)
      return set(stopwords)



def procedural_language(filename):
    stopwords = []
    with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
        stopwords = fp.read().split('\n')
    return set(stopwords)


def normalize_tokens(tokenlist):
    '''
    Input: list of tokens as strings,  e.g. ['I', ' ', 'saw', ' ', '@psresnik', ' ', 'on', ' ','Twitter']
    Output: list of tokens where
    All tokens are lowercased
    All tokens starting with a whitespace character have been filtered out
    All handles (tokens starting with @) have been filtered out
    Any underscores have been replaced with + (since we use _ as a special character in bigrams)
    '''
    normalized_tokens = [token.lower().replace('_','+') for token in tokenlist   # lowercase, _ => +
                             if re.search('[^\s]', token) is not None            # ignore whitespace tokens
                             and not token.startswith("@")                       # ignore  handles
                        ]
    return normalized_tokens

def ngrams(tokens, n):
    '''
    Returns all ngrams of size n in sentence, where an ngram is itself a list of tokens
    '''
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]


def filter_punctuation_bigrams(ngrams):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams
    # Removes ngrams like ['today','.'] where either token is a punctuation character
    # Returns list with the items that were not removed
    punct = string.punctuation
    return [ngram   for ngram in ngrams   if ngram[0] not in punct and ngram[1] not in punct]



def filter_stopword_bigrams(ngrams, stopwords):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams, stopwords is a set of words like 'the'
    # Removes ngrams like ['in','the'] and ['senator','from'] where either word is a stopword
    # Returns list with the items that were not removed
    result = [ngram   for ngram in ngrams   if ngram[0] not in stopwords and ngram[1] not in stopwords]
    return result


contraction_mapping = {
    "ain't": "is not",
    "don't": "do not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "w/": "with",
    "cnt": "cannot",
    "w/o": "without",
    "u": "you"
}




def spacy_preprocessing(text, stopwords, proceduralwords, exclude_list, remove_punctuations, remove_stopwords, remove_nonalpha, remove_procedural):
    '''
    text: accepts stings text
    stopwords: list of stopwords
    proceduralwords: list of procedural words in politics
    exclude_list: Custom list of words to include ex: ['mr','managers']
    clean_tokens: maps words like you're to you are
    returns a clean string

    Parameters
    remove_punctuations: yes removes all puntuations
    remove_stopwords:  yes removes all stopwords
    remove_nonalpha: yes removes all characters execpt uppercase and lowercase letters
    remove_procedural: yes removes all list of procedural words in politics
    Example: text = text = "I am soooooo excited Mr. , to learn nlp. s123 2003 you're doing      great. He will be awesome!!   managers for life"

    '''

    tokens = text.split(" ")
    clean_tokens = [
        contraction_mapping[i] if i in contraction_mapping else i
        for i in tokens
    ]
    # print(clean_tokens)
    text = " ".join(clean_tokens)

    # replaces single random characters in the text with space
    text = re.sub(r"\b([a-zA-Z]{1})\b", " ", text)
    # replaces special characters with spaces
    if remove_nonalpha == 'yes':
        text = re.sub(r"[^a-zA-Z]", " ", text)
    # replaces multiple character with a word with one like pooooost will be post
    text = re.sub(r"(.)\1{3,}", r"\1", text)
    # replaces multiple space in the line with single space
    text = re.sub(r"\s{2,}", r" ", text)

    clean_text = []
    nlp = English(parser=False)
    doc = nlp(text)
    for token in doc:
        if (remove_punctuations == 'yes') & (remove_stopwords == 'yes') & (
                remove_procedural == 'yes'):
            if (token.orth_ not in string.punctuation) & (token.orth_.lower(
            ) not in stopwords) & (token.orth_.lower() not in proceduralwords) & (token.orth_.lower() not in exclude_list):
                clean_text.append(token.orth_.lower())
        elif (remove_punctuations == 'yes') & (remove_stopwords == 'no') & (
                remove_procedural == 'yes'):
            if (token.orth_ not in string.punctuation) & (
                    token.orth_.lower() not in procedural_words):
                clean_text.append(token.orth_.lower())
        elif (remove_punctuations == 'no') & (remove_stopwords == 'yes') & (
                remove_procedural == 'yes') & (token.orth_.lower() not in exclude_list):
            if (token.orth_ not in stopwords) & (
                    token.orth_ not in string.punctuation):
                clean_text.append(token.orth_.lower())
        elif (remove_punctuations == 'yes') & (remove_stopwords == 'yes') & (
                remove_procedural == 'no') & (token.orth_.lower() not in exclude_list):
            if (token.orth_.lower() not in stopwords):
                clean_text.append(token.orth_.lower())
        else:
            clean_text.append(token.orth_.lower())
            continue
    clean_string = " ".join(clean_text).lstrip()

    return clean_string, clean_text

def convert_lines_to_feature_strings_extensive(lines, stopwords, remove_stopword_bigrams=True):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")

    print(" Initializing")
    nlp          = English(parser=False)
    all_features = []
    unigram_features = []
    bigram_features =[]
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):
        #exclude_list=['mr','Mr.','managers','ms','Ms.']
        exclude_list = ['mr','mr.','\'s','n\'t','managers','ms.','a.m.','p.m.','ms','Ms']
        cleaned_strings, cleaned_token = spacy_preprocessing(line, stopwords, None, exclude_list,
                        remove_punctuations='yes', remove_stopwords='yes', remove_nonalpha='yes',
                        remove_procedural='no')
        cleaned_tokens = cleaned_strings.split(' ')
        unigrams =cleaned_tokens

        # Collect string bigram tokens as features
        spacy_analysis    = nlp(line)
        spacy_tokens      = [token.orth_ for token in spacy_analysis]
        normalized_tokens = normalize_tokens(spacy_tokens)

        bigrams = []
        bigram_tokens     = ["_".join(bigram) for bigram in bigrams]
        bigrams           = ngrams(normalized_tokens, 2)
        bigrams           = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
            #bigrams = filter_procedural_bigrams(bigrams, procedural_words)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]


        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'

        # TO DO: replace this line with your code
        feature_unigrams = " ".join(unigrams)
        feature_bigrams = " ".join(bigram_tokens)
        feature_string = unigrams+bigram_tokens
        feature_string= " ".join(feature_string)
        # Add this feature string to the output
        all_features.append(feature_string)
        unigram_features.append(feature_unigrams)
        bigram_features.append(feature_bigrams)


    print(" Feature string for first document: '{}'".format(all_features[0]))

    return all_features, unigram_features, bigram_features




# Emotion Mapping

emotions = pd.read_csv('./word_emotion_intnsity.csv')
def get_word2emotion_mapping():
       '''
       ['Anger', 'Anticipation', 'Disgust', 'Fear','Joy', 'Sadness', 'Surprise', 'Trust']
       sample output: one hot encoded for eemotion
       defaultdict(int,
            {'aback': array([0, 0, 0, 0, 0, 0, 0, 0], dtype=object),
             'abacus': array([0, 0, 0, 0, 0, 0, 0, 1], dtype=object),
             'abandon': array([0, 0, 0, 1, 0, 1, 0, 0], dtype=object),

       '''
       emotion = emotions[['English (en)', 'Anger', 'Anticipation', 'Disgust', 'Fear','Joy', 'Sadness', 'Surprise', 'Trust']]
       word2emotion=defaultdict(int)
       for index,row in emotion.iterrows():
              word=row['English (en)']
              word2emotion[word]=(row[row[row.index].isin([1,0])].values)
       return word2emotion



emotions = pd.read_csv('./word_emotion_intnsity.csv')
def get_word2polarity_mapping():
  '''
  ['Positive', 'Negative']
  sample output: one hot encoded for eemotion
  {'aback': array([0, 0], dtype=object),
             'abacus': array([0, 0], dtype=object),
             'abandon': array([0, 1], dtype=object),

  '''
  emotion = emotions[['English (en)','Positive', 'Negative']]
  word2polarity=defaultdict(int)
  for index,row in emotion.iterrows():
    word=row['English (en)']
    word2polarity[word]=(row[row[row.index].isin([1,0])].values)
  return word2polarity




def word2emotion_matrix(wte, lines, stopwords):
    '''
    Input:
    w2e: Takes a dictionary : word to mapping of an emotion
    ex {'abacus':[0,0,0,0,1,0,0,1]} :  'Anger', 'Anticipation', 'Disgust', 'Fear','Joy', 'Sadness', 'Surprise', 'Trust'
    lines: Takes in the lines of sentencees in an array type.
    stopwords: list of unique stopwords

    returns : emotion matrix for sentence as rows and columns as count of words with a particular emotion in a sentence
    in an array and dataframe format.
    '''
    print("Initializing spacy")
    nlp       = English(parser=False) # faster init with parse=False, if only using for tokenization
    emotion_matrix=[]
    for line in tqdm(lines):
        doc1 = nlp(line)

        spacy_tokens      = [token.orth_ for token in doc1]
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams = [token   for token in normalized_tokens
                                 if token not in stopwords and token not in string.punctuation]
        line_matrix = []
        for token in unigrams:
            if token in wte:
                line_matrix.append(wte[token])
        try:
            summation = sum(np.vstack(line_matrix))
        except:
            summation = np.array([])
        emotion_matrix.append(summation)
        emotion_frame_raw_count= pd.DataFrame(emotion_matrix, columns=['Anger', 'Anticipation', 'Disgust', 'Fear',
           'Joy', 'Sadness', 'Surprise', 'Trust'])
        emotion_frame_normalized_count = emotion_frame_raw_count.div(emotion_frame_raw_count.sum(axis=1), axis=0)
    return emotion_matrix, emotion_frame_raw_count, emotion_frame_normalized_count




def word2polarity_matrix(wtp, lines, stopwords):
    '''
    Input:
    ['Positive', 'Negative']
    sample output: one hot encoded for eemotion
    {'aback': array([0, 0], dtype=object),
               'abacus': array([0, 0], dtype=object),
               'abandon': array([0, 1], dtype=object),
    lines: Takes in the lines of sentencees in an array type.
    stopwords: list of unique stopwords

    returns : emotion matrix for sentence as rows and columns as count of words with a particular emotion in a sentence
    in an array and dataframe format.
    '''
    print("Initializing spacy")
    nlp       = English(parser=False) # faster init with parse=False, if only using for tokenization
    polarity_matrix=[]
    for line in tqdm(lines):
        doc1 = nlp(line)

        spacy_tokens      = [token.orth_ for token in doc1]
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams = [token   for token in normalized_tokens
                                 if token not in stopwords and token not in string.punctuation]
        line_matrix = []
        for token in unigrams:
            if token in wtp:
                line_matrix.append(wtp[token])
        try:
            summation = sum(np.vstack(line_matrix))
        except:
            summation = np.array([])
        polarity_matrix.append(summation)
        polarity_frame_raw_count= pd.DataFrame(polarity_matrix, columns=['positive','negative'])
        polarity_frame_normalized_count = polarity_frame_raw_count.div(polarity_frame_raw_count.sum(axis=1), axis=0)
    return polarity_matrix, polarity_frame_raw_count, polarity_frame_normalized_count


def get_verb_pairs(lines,dep_tag,filter=False):
    '''
    input: Takes in an array of sentences
    returns: verb-dep tag pairs: counts
             verb: counts
    flag: filter==True
         removes if the verb or the dep_tag is a stopword.
    '''
    print("Initializing spaCy")
    verb_objects= defaultdict(int)
    verb_counts=defaultdict(int)
    nlp = spacy.load('en_core_web_sm')
    for row in tqdm(lines):
        doc = nlp(row)
        for token in doc:
                if token.head.pos_ == 'VERB'  and  token.dep_ == dep_tag:
                    x=token.head.lemma_.lower()
                    y=token.lemma_.lower()
                    if dep_tag=='nsubj':
                        verb_object= "_".join([y,x])
                    else:
                        verb_object= "_".join([x,y])
                    verb_counts[x]=verb_counts.get(x,0)+1
                    verb_objects[verb_object]=verb_objects.get(verb_object,0)+1


    if filter:
        filtered_verbs_pairs=defaultdict(int)
        filtered_verbs = defaultdict(int)
        for pair,counts in verb_objects.items():
            x = pair.split('_')
            if (x[0] not in stop_words) and (x[1] not in stop_words):
               filtered_verbs_pairs[pair]=counts
        filtered_verbs_pairs = dict(sorted(filtered_verbs_pairs.items(),key=lambda x:x[1], reverse=True))

        for verb,verb_count in verb_counts.items():
            if verb not in stop_words:
               filtered_verbs[verb]=verb_count
        filtered_verbs = dict(sorted(filtered_verbs.items(),key=lambda x:x[1], reverse=True))
        return filtered_verbs,filtered_verbs_pairs

    verb_counts = dict(sorted(verb_counts.items(),key=lambda x:x[1], reverse=True))
    verb_objects = dict(sorted(verb_objects.items(),key=lambda x:x[1], reverse=True))
    return verb_counts, verb_objects


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

   continious_columns = subset.select_dtypes(include=['float','int']).columns
   if scaler=='minmax':
     mm_scaler = preprocessing.MinMaxScaler()
   elif scaler=='standard':
     mm_scaler = preprocessing.StandardScaler()
   else:
     mm_scaler = preprocessing.RobustScaler()
   subset= mm_scaler.fit_transform(subset)
   return subset, mm_scaler


def split_training_set(lines, labels,test_size=0.3, random_seed=42):
   # TO DO: replace this line with a call to train_test_split
    #X_dev, y_dev = np.array([]), np.array([])
    X_train, X_dev, y_train, y_dev = train_test_split(lines, labels, test_size=.16, stratify=labels, random_state=40)
    #X_train_new, X_test , y_train_new, y_test = np.array([]), np.array([]), np.array([]), np.array([])
    X_train_new,X_test,y_train_new,y_test = train_test_split(X_train, y_train, test_size=.19, stratify=y_train, random_state=40)

    print("Training set label counts: {}".format(Counter(y_train)))
    print("Dev set label counts: {}".format(Counter(y_dev)))
    print("Test set label counts: {}".format(Counter(y_test)))
    return X_train_new, X_test,X_dev, y_train_new, y_test, y_dev


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
