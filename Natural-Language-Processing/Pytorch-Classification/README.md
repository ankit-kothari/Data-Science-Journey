## Code

PyTorch: NLP: Part 6 Multi Class Text Classification with variable sized sequences. 

[Google Colaboratory](https://colab.research.google.com/drive/1f__k2hFhzJhEPisVCs-FgGcYXTgM9BWN?usp=sharing)

## Intended Audience

- Knowledge about basics of PyTorch
- Knowledge about LSTM, RNN's
- Knowledge about word embeddings

## Dataset Cleaning and Extraction

[Data Cleaning, Extraction and Topic Modeling](https://www.notion.so/Data-Cleaning-Extraction-and-Topic-Modeling-0feb04777e85445e806367e6bdd97930)

- The shape of the initial dataset is (2999999, 3) reviews with columns ['rating',' title', 'review']
- Extracted all reviews with the word or mention of Amazon with  (112106, 3)
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

## PyTorch NLP: Text Preprocessing Summary

- Load the data, convert it into a pandas dataframe and followed by a .csv file
- Then using torchtext Filed objects one for the train and one for the label.
    - batch first = True so it (N,T)
    - tokenize using spacy
    - default is string.split(), .split() cannot be used as it is to create tokens as it doesn't understand the punctuations 'dog.' and 'dog' will become two distinct words.
- Next create the TabularDataset object
- Split the dataset using dataset.split
- Call the build vocab function to make the word mapping, vocab.stoi and vocab.itos, pad: 1 , unk: 0
- The Bucketiterator function in torchtext has a parameter sort key to arrange the batches by lenght of sentences, so that we dont have to do heavy padding during batch gradient decent.

pytorch smartly picks up the maxlenght from the batch size itself and not the dataset, for example you data has a maxlen = 20 but the current batch iteration has a sentence of maxlen only 10, pytorch will pad the other sentences only to a max of 10 in that batch. That is a big computation save. 

## Model Architecture

- Create a function RNN and declare the variables and the layers in the __ init __  function.
- Create a forward function which takes in the inputs and applies the layers created in the __ init __  function.
- The function RNN returns an output which is a vector of length K (6 in this case, the number of possible outputs) containing the logits for each outcome.

```python
RNN(
  (dropout1): Dropout(p=0.5, inplace=False)
  (embed): Embedding(20942, 50)
  (rnn): LSTM(50, 128, num_layers=2, batch_first=True)
  (fc): Linear(in_features=128, out_features=6, bias=True)
)
```

```python
class RNN(nn.Module):
  def __init__(self, n_vocab, embed_dim, n_hidden, n_rnnlayers, n_outputs, weights_matrix):
    super(RNN, self).__init__()
    self.V = n_vocab
    self.D = embed_dim
    self.M = n_hidden
    self.K = n_outputs
    self.L = n_rnnlayers
    self.dropout1 = nn.Dropout(0.5)
    self.embedding_matrix = weights_matrix
    self.embed = nn.Embedding(self.V, self.D,    
                         _weight=self.embedding_matrix)

    self.rnn = nn.LSTM(
        input_size=self.D,
        hidden_size= self.M,
        num_layers = self.L,
        batch_first = True
    )
    self.fc = nn.Linear(self.M, self.K)

  def forward(self, X):
    # initial hidden states
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
    c0 = torch.zeros(self.L, X.size(0), self.M).to(device)
    #Embedding
    out = self.embed(X)
    out = out.float()
    #Pass through the LSTM
    out, _ = self.rnn(out,(h0,c0) )
    out = F.relu(out)
    # Max pool
    drop = self.dropout1(out)
    out, _ = torch.max(drop, 1)
    out = F.relu(out)
    #we only want h(T) at the final time step
    out = self.dropout1(out)
    out = self.fc(out)
    return out
```

## Model Output

Train_acc: 97.9286 

Val_accuracy: 95.7106

## Model Predictions

```python
predict(model, "that was a wondeful  movie but the songs were pathetic.")
Output: torch.return_types.max(values=tensor([2.8064], device='cuda:0', grad_fn=<MaxBackward0>), indices=tensor([2], device='cuda:0'))
Correct Prediction: movie
Model Prediction: movie

predict(model, "that was a wondeful  written content by my best author gave me a lot of insights in life. It had a lot of meaing to it.")
torch.return_types.max(values=tensor([1.8185], device='cuda:0', grad_fn=<MaxBackward0>), indices=tensor([0], device='cuda:0'))
Correct Prediction: book
Model Prediction: book
```
