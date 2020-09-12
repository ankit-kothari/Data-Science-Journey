## Code

[Google Colaboratory](https://colab.research.google.com/drive/1w4ffAEqVyKcZm_7uVpRKkvjIRif45Quq?usp=sharing)

## Data Pre-Processing and Exploration

### General terms associated with [spacy](https://spacy.io/usage/linguistic-features) and NLP

- **Text:** The original word text.
- **Lemma:** The base form of the word.
- **POS:** The simple [UPOS](https://universaldependencies.org/docs/u/pos/) part-of-speech tag.
- **Tag:** The detailed part-of-speech tag.
- **Dep:** Syntactic dependency, i.e. the relation between tokens.
- **Shape:** The word shape – capitalization, punctuation, digits.
- **is alpha:** Is the token an alpha character?
- **is stop:** Is the token part of a stop list, i.e. the most common words of the language?
- **Root text:** The original text of the word connecting the noun chunk to the rest of the parse.
- **Root dep:** Dependency relation connecting the root to its head.
- **Root head text:** The text of the root token’s head.
- **Head text:** The original text of the token head.
- **Head POS:** The part-of-speech tag of the token head.
- **Children:** The immediate syntactic dependents of the token.

### **Loading the data**

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-03_at_1.43.42_AM.png" width="100%">

    

### **Data Exploration**

- Grouping the column by Class Name to look at the count by category
- Grouping the column by Clothing ID to look at the count by Clothing ID

```python
**# Count by Class Name** 

Dresses     6319
Knits       4843
Blouses     3097
Sweaters    1428
Pants       1388
Name: Class Name, dtype: int64

**# Count by Clothing ID**

data_dress['Clothing ID'].value_counts()[0:5]

1078    1024
1094     756
1081     582
1110     480
1095     327
Name: Clothing ID, dtype: int64

```

### **Filtering the data**

- To analyze the, filtering it with and **"Dresses"** and **Clothing ID - "1078"**

```python
data_dress = data[(data['Class Name']=='Dresses') & (data['Clothing ID']==1078)]
```
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-03_at_1.48.29_AM.png" width="100%">


### **Converting Reviews into spacy object**

- This step is needed to explore different spacy functions. The NLP object now has a tokenizer, tagger, parser and entity recognizer in its pipeline and we can use it to process a text and get all of those features.

    ```python
    dress_review=data_dress['Review Text'].str.cat(sep='\n')
    doc = nlp(dress_review)
    doc[1200:1600]

    love it! will be easy to wear casually and work appropriate, too. the sale price was a huge bonus.
    I love this dress because its very playful and bouncy. it puts me in a light hearted mood when i wear it. i originally wanted to buy the grey color but my store only had the navy, so i tried it on. the navy is brighter and more colorful than it looks on line and the stripes are more varied in color than in the picture - so its quite appealing and vibrant. the lines of the dress are also quite flattering. all in all, its a fun dress!
    This dress is comfortable as well as flattering, which does not happen very often!
    ```

### **Contents of Spacy NLP pipeline.**

- The Spacy pipeline consist of three parts tagger, parser and ner which are further analyzed below.

    ```python
    [('tagger', <spacy.pipeline.pipes.Tagger at 0x7f18bb185ba8>),
     ('parser', <spacy.pipeline.pipes.DependencyParser at 0x7f18bb05f7c8>),
     ('ner', <spacy.pipeline.pipes.EntityRecognizer at 0x7f18bb05f828>)]
    ```

### Dividing the whole text into **spans** of words, and sentences.

```python
##SPAN
dress_span = doc[0:20]
print(dress_span)
print(type(dress_span))

I really wanted this to work. alas, it had a strange fit for me. the straps would
<class 'spacy.tokens.span.Span'>

##SENTENCES
i=0
for i,sent in enumerate(doc.sents):
    i+=1
    print(sent)
    if i==10:
      break

I really wanted this to work.
alas, it had a strange fit for me.
the straps would not stay up, and it had a weird fit under the breast.
it worked standing up, but the minute i sat down it fell off my shoulders.
the fabric was beautiful!
and i loved that it had pockets.

I love cute summer dresses and this one, especially because it is made out of linen, is unique.
it is very well-made with a design that is quite flattering.
i am 5 foot 6 and a little curvy with a 38 c bust
and i got a size 10.

#To check if token at index 17 is the start of the sentence or not. 

doc[17].is_sent_start
True
```

### POS (Parts of Speech): Understanding the sentence structure

- We could also retrieve some linguistic features such as noun chunks, part of speech tags, and dependency relations between tokens in each sentence. In order to understand what various tags such as token.pos_, token.tag_, or token.dep_ mean, we can use spacy.explain() that will access annotation specifications.
- The entire text(doc) can be sliced with words(token) indices to get single tokens or sequences of tokens (spans) and various token attributes such as text, lemma, index, pos, tag and etc.

    ```python
    for token in doc[0:10]:
        print(f'{token.text:{10}} {token.lemma_:{10}} {token.pos_:{6}} {token.dep_:{12}} {spacy.explain(token.tag_)}')

    I          -PRON-     PRON   nsubj        pronoun, personal
    really     really     ADV    advmod       adverb
    wanted     want       VERB   ROOT         verb, past tense
    this       this       DET    nsubj        determiner
    to         to         PART   aux          infinitival "to"
    ```

- Converting the tokens into pandas DataFrame with **POS** and **LEMMA** for each word

    ```python
    dress_frame = pd.DataFrame()
    o=0
    for token in doc:
        dress_frame.loc[o, 'lemma']= token.lemma_
        dress_frame.loc[o, 'pos']= token.pos_
        dress_frame.loc[o, 'text']= token.text
        dress_frame.loc[o, 'lemma'] = token.dep_
        o=o+1
    dress_frame[0:10]

     lemma	    pos	  text
    0	nsubj	  PRON	  I
    1	advmod	ADV	  really
    2	ROOT	  VERB	wanted
    3	nsubj	  DET	   this
    ```

- Grouping the tokens with **POS (Parts of speech)**

    ```python
    group_dress = dress_frame.groupby(['pos']).agg(
        {
          'text':'count' 
        })
    group_dress['text'].sort_values(ascending=False)[0:15]

    pos
    NOUN     11003
    DET       8329
    PUNCT     8311
    VERB      7634
    PRON      6510
    ADJ       6399
    ADP       5133
    AUX       4879
    ADV       4787
    CCONJ     3339
    PART      1692
    SCONJ     1361
    SPACE     1327
    NUM       1101
    PROPN      822

    ##Getting the TOP 5 Adjectives

    group_dress_adj = dress_frame[dress_frame['pos']=='ADJ'].groupby(['text']).agg(
        {
          'text':'count'      
        })

    group_dress_adj['text'].sort_values(ascending=False)[0:5]

    text
    great          257
    flattering     185
    perfect        176
    comfortable    162
    small          149
    Name: text, dtype: int64
    ```

- Looking at **noun chunks** in the document

    ```python
    #Similar to Doc.ents, Doc.noun_chunks are another object property.
    #Noun chunks are "base noun phrases" – flat phrases that have a noun as their head. 
    #You can think of noun chunks as a noun plus the words describing the noun – for example, 
    #in Sheb Wooley's 1958 song, a *"one-eyed, one-horned, flying, purple people-eater"
    #would be one long noun chunk.
    #https://spacy.io/usage/visualizers

    i=0
    for chunk in doc.noun_chunks:
        i+=1
        print(chunk.text)
        if i ==15:
          break

    I
    it
    a strange fit
    me
    the straps
    it
    a weird fit
    the breast
    it
    i
    it
    my shoulders
    the fabric
    i
    it
    ```

      

- Exploring different **POS tags** across the document for the same word for eg: "size" here

    ```python
    #Text: The original token text.
    #Dep: The syntactic relation connecting the child to head.
    #Head text: The original text of the token head.
    #Head POS: The part-of-speech tag of the token head.
    #Children: The immediate syntactic dependents of the token.

    https://spacy.io/usage/linguistic-features

    i=0
    for token in doc:
      if token.text =="size":
        if i == 15:
          break
        else:
         i+=1
         print(f'{token.text:{14}} {token.head.text:{12}} {token.head.pos_:{10}} {[child for child in token.children]}')

    size           got          VERB       [a, 10]
    size           to           ADP        []
    size           down         ADP        [a]
    size           xs           PROPN      []
    size           xl           PROPN      []
    size           to           ADP        []
    size           was          AUX        [a, petite]
    size           looked       VERB       [neither]
    size           runs         VERB       [so, i, would, down, framed]
    size           purchased    VERB       [the, 4, ,, fit]
    size           small        ADJ        []
    size           needed       VERB       [to, versital, ,, cute]
    size           ordered      VERB       [my, regular, ,, medium]
    size           to           PART       []
    size           returned     VERB       [my]
    ```

### **Name Entity Recognition (NER)**

- Exploring NER Label MONEY
- This will extract all the tokens which are tagged as **"MONEY"** by the Spacy tagger

```python
for ent in doc.ents:
    if ent.label_ == 'MONEY':
     print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))

102# - MONEY - Monetary values, including unit
135 - MONEY - Monetary values, including unit
two cents - MONEY - Monetary values, including unit
128# - MONEY - Monetary values, including unit
140# 34d - MONEY - Monetary values, including unit
120# - MONEY - Monetary values, including unit
140# 5'3 - MONEY - Monetary values, including unit
15 bucks - MONEY - Monetary values, including unit
158 - MONEY - Monetary values, including unit
120# - MONEY - Monetary values, including unit
over $50 - MONEY - Monetary values, including unit
5'3 - MONEY - Monetary values, including unit
110# - MONEY - Monetary values, including unit
168 - MONEY - Monetary values, including unit
49 - MONEY - Monetary values, including unit
79 - MONEY - Monetary values, including unit
39;fuzzy&#39 - MONEY - Monetary values, including unit
over $250 - MONEY - Monetary values, including unit
#32b# - MONEY - Monetary values, including unit
120# max - MONEY - Monetary values, including unit
138 - MONEY - Monetary values, including unit
135# 36c - MONEY - Monetary values, including uni
```

- **Retrieving and visualizing named entities** is done very conveniently in spaCy.

```python
displacy.render(doc[0:500], style='ent', jupyter=True, options={'distance': 110})
```

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-03_at_2.26.12_AM.png" width="80%">

### Adding **Custom Name Entity Tags** to the document

- I want to tag the word "dress" as a "PRODUCT" in the entire dataset

```python
for i,token in enumerate(doc):
  if token.text == 'dress':
     print(token.text)
     print(i)
     break

Output:
dress
140

new_ent = Span(doc,140, 141, label=PRODUCT)
doc.ents = list(doc.ents)+ [new_ent]
```

- Now the word dress is tagged and can be seen when we filter for **"PRODUCTS"** in the document

```python
**dress - PRODUCT - Objects, vehicles, foods, etc. (not services)**
the s fit - PRODUCT - Objects, vehicles, foods, etc. (not services)
the s fit great - PRODUCT - Objects, vehicles, foods, etc. (not services)
p6 - PRODUCT - Objects, vehicles, foods, etc. (not services)
s - PRODUCT - Objects, vehicles, foods, etc. (not services)
a34b - PRODUCT - Objects, vehicles, foods, etc. (not services)
small/ - PRODUCT - Objects, vehicles, foods, etc. (not services)
```

## Stemming (not included in spacy and only used in NLTK)

- Stemming on ADJECTIVE's
- The Stemming is the process of reducing the word into it's root form.

```python
dress_adj = dress_frame[(dress_frame['pos']=='ADJ') & (dress_frame['lemma']=='acomp' )]
dress_adj[0:10]

         	pos	text
54	acomp	ADJ	beautiful
84	acomp	ADJ	unique
98	acomp	ADJ	flattering
127	acomp	ADJ	difficult
181	acomp	ADJ	nice
200	acomp	ADJ	true
229	acomp	ADJ	lovely
253	acomp	ADJ	perfect
287	acomp	ADJ	adorable
292	acomp	ADJ	flattering

The stem of the word beautiful  is beauti              
The stem of the word unique     is uniqu               
The stem of the word flattering is flatter             
The stem of the word difficult  is difficult           
The stem of the word nice       is nice                
The stem of the word true       is true                
The stem of the word lovely     is love                
The stem of the word perfect    is perfect             
The stem of the word adorable   is ador                
The stem of the word worse      is wors
```

- Stemming on VERB's

```python
     	pos	text
91	  acomp	VERB	made
567	  acomp	VERB	wearing
2526	acomp	VERB	made
4471	acomp	VERB	chested
4853	acomp	VERB	faded
6134	acomp	VERB	closed
8687	acomp	VERB	worried
9350	acomp	VERB	dressed
10675	acomp	VERB	looking
10781	acomp	VERB	endowed

The stem of the word made       is made                
The stem of the word wearing    is wear                
The stem of the word chested    is chest               
The stem of the word faded      is fade                
The stem of the word closed     is close               
The stem of the word worried    is worri               
The stem of the word dressed    is dress               
The stem of the word looking    is look                
The stem of the word endowed    is endow               
The stem of the word pictured   is pictur

```

## Lemmatization

1. In contrast to stemming, lemmatization looks beyond word reduction, and considers a language's full vocabulary to apply a morphological analysis to words.

2. The lemma of 'was' is 'be' and the lemma of 'mice' is 'mouse'.

3. Further, the lemma of 'meeting' might be 'meet' or 'meeting' depending on its use in a sentence.

- Lemmatization on ADJECTIVE's

```python
i=0
for token in doc:
    if token.pos_ == 'ADJ' and token.dep_ =='acomp':
      if i<10:
        i+=1
        print(f'{token.text:{10}} {token.lemma_:{10}} {token.pos_:{6}} {token.dep_:{12}} {spacy.explain(token.tag_)}')
      else:
        break

beautiful  **beautiful**  ADJ    acomp        adjective
unique     **unique**     ADJ    acomp        adjective
flattering **flattering** ADJ    acomp        adjective
difficult  **difficult**  ADJ    acomp        adjective
nice       **nice**       ADJ    acomp        adjective
true       **true**       ADJ    acomp        adjective
lovely     **lovely**     ADJ    acomp        adjective
perfect    **perfect**    ADJ    acomp        adjective
adorable   **adorable**   ADJ    acomp        adjective
flattering **flattering** ADJ    acomp        adjective
```

- Lemmatization on VERB's

```python
i=0
for token in doc:
    if token.pos_ == 'VERB' and token.dep_ =='acomp':
      if i<10:
        i+=1
        print(f'{token.text:{10}} {token.lemma_:{10}} {token.pos_:{6}} {token.dep_:{12}} {spacy.explain(token.tag_)}')
      else:
        break

made       **make**       VERB   acomp        verb, past participle
wearing    **wear**       VERB   acomp        verb, gerund or present participle
made       **make**       VERB   acomp        verb, past participle
chested    **cheste**     VERB   acomp        verb, past participle
faded      **fade**       VERB   acomp        verb, past participle
closed     **close**      VERB   acomp        verb, past participle
worried    **worry**      VERB   acomp        verb, past participle
dressed    **dress**      VERB   acomp        verb, past participle
looking    **look**       VERB   acomp        verb, gerund or present participle
endowed    **endow**      VERB   acomp        verb, past participle
```

## Stop Words

- List of Default Stop Words

```python
stop_words = nlp.Defaults.stop_words
stop_word = [i for i in stop_words]
stop_word[0:10]

['during',
 'never',
 'besides',
 'thereafter',
 'since',
 'or',
 'noone',
 'rather',
 'often',
 'though']
```

- Check if the word is a stop-words or not

```python
nlp.vocab['is'].is_stop
True

nlp.vocab['mystery'].is_stop
False
```

- Adding a stop word to the list of default list of stop-words

```python
#Adding a stop word
#Add the word to the set of stop words. Use lowercase!
nlp.Defaults.stop_words.add('btw')

# Set the stop_word tag on the lexeme
nlp.vocab['btw'].is_stop = True
```

- Removing a stop word from the list of default list of stop-words

```python
# Remove the word from the set of stop words
nlp.Defaults.stop_words.remove('however')

# Remove the stop_word tag from the lexeme
nlp.vocab['however'].is_stop = False

nlp.vocab['however'].is_stop
False
```

## Vocabulary and Matching

**There are two ways of matching text in spacy, Below are the following**

- Matcher

```python
pattern1 = [{'LOWER': 'tight','OP':'+'}]
pattern2 = [{'LOWER': 'petites'}]
matcher.add('sizes', None, pattern1,pattern2)
matcher

i found the fit to be flattering -- fitted enough but not too loose or **tight**.
i do think the cut is on the trim side, but it isn't **tight** or fitted.
it is a bit looser on top (i'm 32c) and more form-fitting around the hips but not **tight** or clingy.
the end of the sleeves (where the buttons are) are very **tight** but
the arm holes were **tight**, but have very cute buttons if you look closely at the picture.  
both fit well, but the sleeves in the large were quite **tight**.
when it arrived and i tried it on, it fit great on my arms, wasn't too tight on my neck, but once it went down over my chest (36c), the dress never came back "in" to show my feminine waist/shape.it is a perfect length and drapes well...not too **tight** at all.
ok, the arms were a little **tight**.
i also love how versatile it is--you can wear it as a dress as shown on the models, or as a tunic over **tight** jeans or jeggings/leggings.
```

- PhraseMatcher

```python
phrase_matcher = PhraseMatcher(nlp.vocab)
phrase = ['tight', 'extremely tight', 'too tight', 'wrong size']
phrase_patterns =  [nlp(text) for text in phrase]
print(phrase_patterns)
phrase_matcher.add('rsk', None, *phrase_patterns)

the armholes fit perfectly though, if i had sized down they may have been **too tight**.
the slip was also slightly **tight** over hips.
i may have to try, though, as otherwise it's a great workable dress for summer--not **too tight** or revealing in the b
i may have to try, though, as otherwise it's a great workable dress for summer--not **too tight** or revealing in the b
for myself, the sleeves are **tight** and  so is the fit across the back and shoulders.  
one comment mentioned that the slip underneath was **tight** in the hips, yet as a size 30 in pants the slip was still fine.
This dress is beautiful but the bottom half was **too tight** for my shape.
This dress is beautiful but the bottom half was **too tight** for my shape.
```

## Sentence Segmentation

- Default Segmentation rule.

```python
print(nlp.pipe_names)
['tagger', 'parser', 'ner']
```

- After Adding custom segmentation rule.
- So by default a new sentence ends with **"."** , but what if we want to end the sentence with **","** as in case of poems.

```python
def set_custome_segmentations(doc):
  for token in doc[:-1]:
    if token.text == ",":
      doc[token.i+1].is_sent_start = True
  return doc

nlp.pipe_names
nlp.add_pipe(set_custome_segmentations, before='parser')
print(nlp.pipe_names)

['tagger', 'set_custome_segmentations', 'parser', 'ner']
```

- Before Segmentation Rule.

```python
for sent in dec_ss.sents:
  print(sent)

**I really wanted this to work..
alas, it had a strange fit for me..
the straps would not stay up, and it had a weird fit under the breast.**
```

- After Adding custom segmentation rule.
- Now the new sentence ends with **","**

```python
doc_post_ss = nlp(ss)
for sent in doc_post_ss.sents:
  print(sent)

**I really wanted this to work..
alas,
it had a strange fit for me..
the straps would not stay up,
and it had a weird fit under the breast.**
```
