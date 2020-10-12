
## Transfer Learning using Transformer Based Model

### distilBERT
- the Hugging Face library seems to be the most widely accepted and powerful pytorch interface for working with distilBERT. In addition to supporting a variety of different pre-trained language models (and future models to come - distilBERT will not be state of the art forever), the library also includes pre-built modifications of distilBERT suited to your specific task. For example, in this tutorial we will use BertForSequenceClassification, but the library also includes distilBERT modifications designed for token classification, question answering, next sentence prediciton, etc. Using these pre-built classes simplifies the process of modifying distilBERT for your purposes.


### distilBERT requires specifically formatted inputs. For each tokenized input sentence, we need to create:

- [**input ids**]: a sequence of integers identifying each input token to its index number in the distilBERT tokenizer vocabulary
- [**segment mask**]: (optional) a sequence of 1s and 0s used to identify whether the input is one sentence or two sentences long. For one sentence inputs, this is simply a sequence of 0s. For two sentence inputs, there is a 0 for each token of the first sentence, followed by a 1 for each token of the second sentence
- [**attention mask**]: (optional) a sequence of 1s and 0s, with 1s for all input tokens and 0s for all padding tokens (we'll detail this in the next paragraph)
- [**labels**]: a single value of 0,1,2,3,4,5. In our task it means ['books', 'games', 'movies', 'music', 'refund-and-return',
       'video-quality']
       
       
 ### Using the pre Trained Encoder Output, passing it through a classificatio model as features to classify
 - Using a simple ANN network created a Multi classification model using the features from the pre-trained a distilBERT library 
   
 
