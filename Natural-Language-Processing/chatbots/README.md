### Code

[ankit-kothari/chatbots](https://github.com/ankit-kothari/chatbots)

## Intended Audience

- People who have basic knowledge of NLP text preprocessing
- Working knowledge of word embeddings and TF-IDF algorithms for text to vector represntation
- Knowledge of how to utilitze TF hub to get the sentence encoder  embeddings from Tensorflow 2.0
- Knowledge  of Slack API

## Chatbot Introduction

FAQ chatbots are common form of chatbots but can be of great utility to address basic questions and be used in variety of forms:

- Customer Service
- Documentation of Products
- Product/Company Policy
- General Business Operations questions

The different methods we used in this article to explore accuracy  are:

- TFIDF with no stop words.
- TFIDF with stop words.
- Glove Word Embeddings with 100 dimensions.
- Tensorflow Endoder Dimension with 512 dimensions.

## Success Criteria

All of the above methods are tested on each of the below questions. The questions are choosesn in a way so that we evaluate how the chatbot functions when it sees any unknown word, whether it understands similar meanings (semantics)

- How much degree should be for good brew?
- Can coffee expire?
- How many cups can i have in a day?
- Which country produces the most amount of coffee in the world?
- How to clean coffee on my shirt?

## Create an app in Slack

- Click on the APP icon in slack to add apps.
- Search for bots and configure it with username and basic details.
- This will give you a secret token 'x...-....'

## Connecting the script to Slack

```python
slack_clients = slackclient.SlackClient('secret-token')
```

## Dataset

- I have manually curated the dataset which has basic FAQ questions about coffee. It consist of 27 questions with a unique answer each. The top 5 rows of the dataset are as follows.

[Coffee Dataset](https://www.notion.so/30d689e48980486ab7bdd54c96d6684e)

## Text Preprocessing

- NLP Terminologies: Non-Text characters, Stop-words, Stemming, Lemmatization
- We used NLTK to clean the sentences aremove all the stop words
- We also remove all the symbols other than texts.
- Converting it to lower case
- Converting the token to its Lemmatized form

Without stop words

```
0    what best temperature brew coffee             
1    quality coffee                                
2    what difference arabica robusta               
3    just much ground coffee i need x amount coffee
4    what different preparation methods            
Name: nltk_cleaning, dtype: object
```

- We also remove all the symbols other than texts.
- Converting it to lower case.
- Converting the token to its Lemmatized form

With stop words

```
0    what be the best temperature to brew coffee                 
1    quality of coffee                                           
2    what be the difference between arabica and robusta          
3    just how much ground coffee do i need for x amount of coffee
4    what be the different between preparation methods           
Name: nltk_cleaning, dtype: object
```

## Text Vectorization

### TF-IDF

- **Term Frequency**: is a scoring of the frequency of the word in the current document.

    ```
    TF = (Number of times term t appears in a document)/(Number of terms in the document)
    ```

- **Inverse Document Frequency**: is a scoring of how rare the word is across documents.

    ```
    IDF = 1+log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
    ```

- Tf-IDF weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus

### Glove Word Embeddings

- Global Vectors for words representation (GloVe) is provided by Stanford. They provided various models from 25, 50, 100, 200 to 300 dimensions based on 2, 6, 42, 840 billion tokens
- Team used word-to-word co-occurrence to build this model. In other words, if two words co-occur many times, it means they have some linguistic or semantic similarity.
- I have used the embeddings with 100 dimension

```python
EMBEDDING_DIM = 100

# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('./glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))
```

for more detailed explanation, one good article,  [https://towardsdatascience.com/art-of-vector-representation-of-words-5e85c59fee5](https://towardsdatascience.com/art-of-vector-representation-of-words-5e85c59fee5)

### Tensorflow Sentence Encoder

- The model constructs sentence embeddings using the encoding sub-graph of the transformer architecture. The encoder uses attention to compute context-aware representations of words in a sentence that take into account both the ordering and identity of other words. The context-aware word representations are averaged together to obtain a sentence-level embedding.
- This code will create a sentence embedding for all the 27 questions with each being of 512 dimension

```python
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
que_list = [embed([x]) for x in data['nltk_cleaning']]
que_array = np.array(que_list)
que_array=que_array.reshape(27,512)
```

## Parsing incoming text from Slack (user input)

- Whenever the user input is entered, it is parsed by Slack API as  event['type']="message"  and this is passed to handle_command function (scroll down in the article) to create a response.

```python
def parse_bot_commands(slack_events):
    for event in slack_events:
         if event["type"] == "message" and not "subtype" in event:
              message = event["text"]
              return message, event["channel"], event['user']
    return None, None, None
```

## Creating response and posting back in Slack

- Access Control: Replace  '<SLACK USER ID>' with a list of accepted Slack User ID's
- Create an exit criteria.
- Cosine similarity of any pair of vectors by taking their dot product and dividing that by the product of their norms. That yields the cosine of the angle between the vectors. Cosine similarity is a measure of similarity between two non-zero vectors. Using this formula we can find out the similarity between any question to rest of the question dataset.
- Find the index of the highest matched question from the dataset.
- Return the corresponding Response.
- Post it the chatbot in Slack using the Slack API

```python
def handle_command(command, channel,user):
    # Default response is help text for the user
    accepted_users = ['<SLACK USER ID>']
    ending_text = ['bye','done', 'thanks', 'exit', 'ok', 'x']
    if user not in accepted_users:
          response = "Not an authorised user"
    elif command in ending_text:
          response = "Bye! Thanks for chatting"
    else:
      try:
        Question_lemma_tf = nltk_cleaning(command) # applying the function that we created for text normalizing
        Question_tf = tfidf.transform([Question_lemma_tf]).toarray() # applying bow
        cosine_value_tf = 1- pairwise_distances(df_idf, Question_tf, metric = 'cosine' )
        index_value = cosine_value_tf.argmax()
        response = data['Text Response'].loc[index_value]
      except:
          response='Sorry, Not sure what you mean'
        
    

    # Sends the response back to the channel
    slack_clients.api_call(
        "chat.postMessage",
        type="divider",
        channel=channel,
        #text=':coffee:'+':coffee:'+"*"+response+"*" + ':coffee:'+':coffee:',
        attachments=[{
        "blocks":[
		{
			"type": "section",
			"block_id": "section567",
			"text": {
				"type": "mrkdwn",
				"text": ':coffee:'+':coffee:'+"*"+response+"*" + ':coffee:'+':coffee:'
			},
			"accessory": {
				"type": "image",
				"image_url": "https://png.pngtree.com/png-clipart/20190903/original/pngtree-yellow-woven-bag-of-coffee-beans-png-image_4418658.jpg",
				"alt_text": "Haunted hotel image"
			}}
		]}]
    )

##Running the application
print("Done")
```

## Running the Slack Chatbot

- The slack_clients.rtm_connect() function connects the script to the Slack bot App and it is constantly waiting to read the 'user input' .
- If the user input is 'shutdownthebot' the Slack bot is disconnected
- the user input is analyzed by the parse_bot_commands(slack_clients.rtm_read())
- based on the input  from an accepted user handle_command(command, channel, user) creates a response which is posted back in Slack as response.

```python
# constants
RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
if __name__ == "__main__":
    if slack_clients.rtm_connect(with_team_state=False):
        print("Starter Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        starterbot_id = slack_clients.api_call("auth.test")["user_id"]
        while True:
            command, channel, user = parse_bot_commands(slack_clients.rtm_read())
            if command == 'shutdownthebot':
                break
            else:
               if command is None:
                  continue
               else:
                  handle_command(command, channel, user)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")
```

Output

```
Starter Bot connected and running!
```

## Results

The observations as seen by the results from all the same 5 questions 

- TF-IDF with stopwords created a feature matrix which weighed the stop words which are too common in the corpus and resulted in bad results.
- TF-IDF without stopwords/ word embeddings did well where the words of the questions were present in the corpus, where as they didn't do well where the words in corpus were replaced by symanctic meanings.
- TF sentence encoder worked the best since its encoding the sentence using attention mechanism understanidng semantics and can relate to similar word meanings and the order of words in a sentence.


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/IMG_0029.jpg" width="50%">


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/IMG_0028.jpg" width="50%">

## Sources

- [https://www.cafepoint.co.uk/blog/04-12-2015/the-20-most-asked-coffee-questions/](https://www.cafepoint.co.uk/blog/04-12-2015/the-20-most-asked-coffee-questions/)
- [https://coffee-brewing-methods.com/coffee/what-is-coffee/](https://coffee-brewing-methods.com/coffee/what-is-coffee/)
- [https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d#2bd3](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d#2bd3)
