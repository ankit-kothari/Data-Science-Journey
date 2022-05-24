# Deep-Learning
## Why are shapes important in Deep Learning?

I think understanding shapes in Deep Learning is the most fundamental to build anything comprehensive and innovative. There is a lot out there where we can cut-copy-paste and it will work fine but for more complex problems or innovations or when you have to apply the knowledge of deep learning to your dataset, this knowledge will be of great importance to build your layers, understand where the issues are and even during hyperparameter tuning.

## The audience for this article:

- People who have theoretical understanding of what different architectures like RNN, LSTM, GRU and CNN are used for and how they typically fit with sequence data or images.

## Experimenting with different Architectures

- FeedForward Network
- Recurrent Nueral Network (RNN)
- Long Short Term Memory Network (LSTM)
- Bi-Directional LSTM (Bi-LSTM)
- Gated Recurrent Units (GRU's)
- Convolutional Neural Network

## Key Definations:

T = Sequence Length

- sequence can be a length a sentence in a dataset
- sequence can be one years of sales data(by month)  in a dataset,

D = Embedding Dimension

- The word gets converted into a vector using glove or word2vec and it can be of dimension 50,100, 300

M = Hidden Units 

- Hidden units are used by RNN, LSTM, GRU's

N = Batch Size 

- Number of samples that needs to be trained in one iteration (different from epoch)/

one epoch is a complete pass of all the training data through the model, there can be n iterations depending upon the batch size. **example: we can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch.
Where Batch Size is 500 and Iterations is 4, for 1 complete epoch.**

**example using NLP sentences:**

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/deep_learning_embedding.png" width="40%">


## Declarations:

Creating a random input variable X with Batch size, N = 1 and sequence Length T=8, and vector length dimension D = 2.

```python
T = 8   ### T is the sequnces lenght
D = 2   ### D is the vector length dimensionality
M = 3   ### hidden layer size

X = np.random.randn(1, T, D)
```

```
array([[[-2.04695422,  0.91626883],
        [-0.54788076, -1.60473076],
        [-0.28452751,  0.9593858 ],
        [ 0.45396515,  1.4244805 ],
        [ 1.25445189,  0.16370936],
        [ 0.46161679, -1.21975855],
        [-1.41926118,  0.70528338],
        [-0.24266939,  1.16912058]]])
```

## FeedForward Network:

- In the following feedforward we have 3 Dense neurons so all the 8 (T= sequence length) gets multiplied to each of the neurons and produces 24 parameters in the output.

```python
def Feedforward():
  input_ = Input(shape=(T, D))
  rnn = Dense(M, activation='sigmoid')
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  model.summary()
  o= model.predict(X)
  print("Feedforward output:", o)
  print("Feedforward output.shape:", o.shape)
Feedforward()
```

Output:

```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 8, 2)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 8, 3)              9         
=================================================================
Total params: 9
Trainable params: 9
Non-trainable params: 0
_________________________________________________________________
Feedforward output: 
[[[0.6705187  0.46951422 0.40324914]
  [0.6971361  0.13498545 0.7961645 ]
  [0.45247954 0.69078135 0.32861826]
  [0.32656258 0.8365394  0.23073679]
  [0.33454144 0.68788016 0.417011  ]
  [0.5478986  0.27450985 0.7093028 ]
  [0.61498237 0.49609575 0.41890526]
  [0.4293114  0.73723495 0.2915951 ]]]
Feedforward output.shape ***(NxTxM)*** : (1, 8, 3)
```

## Recurrent Neural Network (RNN):

- IN RNN if we have 3 (M=3 Hidden Units) when we pass in a sequence of T=8, it goes one by one into the network
- When "return_state=True" alone, It will output the final value at T(7) (timestamp (0-7) one from each hidden neuron and the shape will be (1,3) and since return states are true is will also output h: the final hidden state for each of the hidden neuron (1,3)

```python
def RNN1():
  input_ = Input(shape=(T, D))
  rnn = SimpleRNN(M,return_state=True )
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  model.summary()
  o, h= model.predict(X)
  print("RNN o:", o)
  print("RNN o.shape:", o.shape)
  print("RNN h:", h)
  print("RNN h:", h.shape)
RNN1()
```

Output:

```
Model: "model_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         (None, 8, 2)              0         
_________________________________________________________________
simple_rnn_4 (SimpleRNN)     [(None, 3), (None, 3)]    18        
=================================================================
Total params: 18
Trainable params: 18
Non-trainable params: 0
_________________________________________________________________
RNN o: [[ 0.9291756  -0.79711425 -0.38878286]]
RNN o.shape **(N,M)** : (1, 3) 
RNN h: [[ 0.9291756  -0.79711425 -0.38878286]]
RNN h **(N,M)**: (1, 3)
```

- When if the "return_sequneces" parameter is True as well, then we will get an output vector at each timestamp from T(0) to T(7) (8 is the sequence length) so T1 will have (1,3) from each hidden state all the way up to T(7) making the output shape to be (8,3)

```python
def RNN2():
  input_ = Input(shape=(T, D))
  rnn = SimpleRNN(M,return_state=True, return_sequences=True )
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  model.summary()
  o, h= model.predict(X)
  print("RNN3 o:", o)
  print("RNN3 o.shape:", o.shape)
  print("RNN3 h:", h)
  print("RNN3 h:", h.shape)
RNN3()
```

Output:

```
Layer (type)                 Output Shape              Param #   
=================================================================
input_16 (InputLayer)        (None, 8, 2)              0         
_________________________________________________________________
simple_rnn_6 (SimpleRNN)     [(None, 8, 3), (None, 3)] 18        
=================================================================
Total params: 18
Trainable params: 18
Non-trainable params: 0
_________________________________________________________________
RNN2 o: [[[-0.10498122  0.06420284 -0.26365352]
  [ 0.65598315  0.20150483  0.2275719 ]
  [-0.2999473  -0.3333473  -0.5453389 ]
  [ 0.19372658  0.03814533  0.65649503]
  [ 0.2869922   0.54233384 -0.6235938 ]
  [ 0.8993942   0.48097286 -0.90915865]
  [ 0.7897056  -0.5968131  -0.10384333]
  [ 0.15055196  0.8274371  -0.9671663 ]]]
RNN2 o.shape **(NxTxM)**: (1, 8, 3)
RNN2 h: [[ 0.15055196  0.8274371  -0.9671663 ]]
RNN2 h **(N,M)**: (1, 3)
```

The output at T(7) and the hidden state values are the same 

## Long Short Term Memory Network (LSTM)

They are used often as compared to RNN’s as they are better suited for long term dependencies and vanishing gradient problems in RNN. three outputs from an LSTM layer namely output (o), the hidden state (h1), and the cell state (c1).

- With "return_state=True" alone
- state refers to hidden state (h1) and cell state(c1) for LSTM

```python
def lstm1():
  input_ = Input(shape=(T, D))
  rnn = LSTM(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  model.summary()
  o, h1, c1= model.predict(X)
  print("o:", o)
  print("o.shape:", o.shape)
  print("h1:", h1)
  print("h1:", h1.shape)
  print("c1:", c1)
  print("c1:", c1.shape)
lstm1()
```

```
Model: "model_14"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_17 (InputLayer)        (None, 8, 2)              0         
_________________________________________________________________
lstm_4 (LSTM)                [(None, 3), (None, 3), (N 72        
=================================================================
Total params: 72
Trainable params: 72
Non-trainable params: 0
_________________________________________________________________
o: [[ 0.01168317  0.12095464 -0.2356035 ]]
o.shape **(NxM)**: (1, 3)
h1: [[ 0.01168317  0.12095464 -0.2356035 ]]
h1 **(NxM)**: (1, 3)
c1: [[ 0.02696911  0.20223528 -0.44894364]]
c1 **(NxM)**: (1, 3)
```

- With "return_state=True" and "return_sequneces" parameter is True as well,

```python
def lstm2():
  input_ = Input(shape=(T, D))
  rnn = LSTM(M, return_state=True, return_sequences=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  model.summary()
  o, h1, c1= model.predict(X)
  print("o:", o)
  print("o.shape:", o.shape)
  print("h1:", h1)
  print("h1:", h1.shape)
  print("c1:", c1)
  print("c1:", c1.shape)
lstm2()
```

```
Model: "model_15"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_18 (InputLayer)        (None, 8, 2)              0         
_________________________________________________________________
lstm_5 (LSTM)                [(None, 8, 3), (None, 3), 72        
=================================================================
Total params: 72
Trainable params: 72
Non-trainable params: 0
_________________________________________________________________
o: [[[ 0.0668667   0.00764452 -0.0018249 ]
  [-0.02357746 -0.03809585  0.06874069]
  [ 0.02714967  0.00278267  0.01081553]
  [ 0.00202328  0.01083693 -0.01706302]
  [-0.0461082  -0.04218157  0.08063943]
  [ 0.02728271 -0.04221687  0.08875341]
  [-0.03200685 -0.03021823  0.07329794]
  [ 0.21305892  0.00563298  0.03126976]]]
o.shape **(NxTxM)**: (1, 8, 3)
h1: [[0.21305892 0.00563298 0.03126976]]
h1 **(NxM)**: (1, 3)
c1: [[0.28252053 0.01348653 0.09836657]]
c1 **(NxM)**: (1, 3)
```

## Bi-Directional LSTM (Bi-LSTM)

In this architecture we the sequence once from T(0) to T(7) and then we start from T(7) to T(1). This proves really helpful in remembering long term dependencies.

```python
def bidirectional():
 input_ = Input(shape=(T, D))
 rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=True))
 x = rnn(input_)

 model = Model(inputs=input_, outputs=x)
 o, h1, c1, h2, c2 = model.predict(X)
 print("o:", o)
 print("o.shape:", o.shape)
 print("h1:", h1)
 print("h1:", h1.shape)
 print("c1:", c1)
 print("c1:", c1.shape)
 print("h2:", h2)
 print("c2:", c2)
bidirectional()
```

```
o: [[[-0.0511428   0.00461203 -0.00480167  ***0.05690131 -0.04843732
    0.0173745*** ]
  [ 0.04451018 -0.07120016  0.02538789 -0.01117533 -0.0792519
    0.06766772]
  [ 0.01125101 -0.0055525  -0.00213479  0.08502878 -0.04543929
   -0.03407311]
  [ 0.01788764  0.02942917 -0.00855108  0.04438303 -0.13224526
    0.02475515]
  [ 0.07994339 -0.07786338  0.0512228   0.04396992 -0.18825892
    0.05643139]
  [-0.07597981 -0.15106945  0.12810288  0.1280036  -0.09966266
   -0.08037673]
  [-0.00921727 -0.12368339  0.03524554  0.11016312 -0.02101769
   -0.08303176]
  [**-0.2683192  -0.11089069 -0.00252563**  0.12397347 -0.02505051
   -0.13302208]]]
o.shape **(NxTx2M)**: (1, 8, 6)
h1: [[***-0.2683192  -0.11089069 -0.00252563***]]
h1 **(NxM)**: (1, 3)
c1: [[-0.41002703 -0.20509982 -0.00386132]]
c1 **(NxM)**: (1, 3)
h2: [[ ***0.05690131 -0.04843732  0.0173745*** ]]
h2 **(NxM)**: (1, 3)
c2: [[ 0.12594512 -0.10560355  0.03287323]]
c2 **(NxM)**: (1, 3)
```

the last 3 values at T(0) is equal to the values of h1 and the first three values at T(7) are equal to h2.

## Gated Recurrent Units (GRU's)

They are used often as compared to RNN’s as they are better suited for long term dependencies and vanishing gradient problems in RNN, two output from a GRU layer namely output(o) and the hidden state (h). Their performance are more or similar to LSTM's. 

- With "return_state=True" alone
- state refers to hidden state (h)  for GRU

```python
def gru1():
  input_ = Input(shape=(T, D))
  rnn = GRU(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h = model.predict(X)
  print("o:", o)
  print("h:", h)
gru1()
```

```
o: [[ 0.12817755 -0.5492754  -0.43603235]]
h: [[ 0.12817755 -0.5492754  -0.43603235]]
```

- With "return_state=True" and "return_sequneces" parameter is True as well,

```python
def gru2():
  input_ = Input(shape=(T, D))
  rnn = GRU(M, return_state=True, return_sequences=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  model.summary()
  o, h = model.predict(X)
  print("o:", o)
  print("h:", h)
gru2()
```

```
Model: "model_17"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_20 (InputLayer)        (None, 8, 2)              0         
_________________________________________________________________
gru_4 (GRU)                  [(None, 8, 3), (None, 3)] 54        
=================================================================
Total params: 54
Trainable params: 54
Non-trainable params: 0
_________________________________________________________________
o: [[[-0.05129799 -0.12828992 -0.05382598]
  [ 0.14443547  0.06542104  0.05674175]
  [-0.05337665 -0.04498286  0.0083748 ]
  [-0.05412816  0.00636416  0.01542265]
  [ 0.17073159  0.10832258  0.07074815]
  [ 0.18945819 -0.1922195  -0.06368452]
  [ 0.04970717 -0.0034977   0.01780862]
  [-0.13356349 -0.52509594 -0.23108722]]]
h: [[-0.13356349 -0.52509594 -0.23108722]]
```

## Convolutional Neural Network

```python
def convert_images(imagetensor):
  tensor2array = imagetensor.clone().detach().numpy()
  tensor2array = tensor2array.transpose(1,2,0)
  print((f' tensor2array shape {tensor2array.shape}'))
  tensor2array = tensor2array * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
  print(f'transposed array {tensor2array.shape}')
  return tensor2array.clip(0,1)

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg'
response = requests.get(url, stream = True)
img1 = Image.open(response.raw)
transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                               ])

img1  = convert_images(transform(img1))
catbatch = np.expand_dims(img1 , axis=0)
print(f' image type {type(img1)}')
print(f' image shape {(img1.shape)}')
print(f' catbatch type {catbatch.shape}')
print(f' catbatch type {type(catbatch)}')
catbatch = np.expand_dims(img1, axis=0)
print(f' expanded dims image shape {catbatch.shape}')
plt.imshow(img1)
```
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cat_before.png" width="50%">

original image

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cat_after.png" width="50%">

pre-processed image through pytorch

```
tensor2array shape (28, 28, 3) (**H x W x ColorChannel)**
transposed array (28, 28, 3) (**H x W x ColorChannel)**
image type <class 'numpy.ndarray'>
image shape (28, 28, 3) (**H x W x ColorChannel)**
catbatch type (1, 28, 28, 3) **(N x H x W x ColorChannel)**
catbatch type <class 'numpy.ndarray'>
expanded dims image shape (1, 28, 28, 3) **(N x H x W x ColorChannel)**
```

This is a very simple model with just one layer of Convolution and with a kernal (filter) size of (2x2)

```python
model = Sequential()
model.add(Conv2D(32,kernel_size=2))
conv_cat = model.predict(catbatch)
conv_cat.shape
```

**Output:**

```
(1, 27, 27, 32)
```

---

Now,

Here it should be noted that the output from the PyTorch model is in the form (Channel x Height x Width x Number of kernels(feature maps) but Matplotlib expects it to be (Nx Height x Width). This is what the following code is doing.

Every kernel(filter) creates a new image, so we have a stack of filtered images (32) in this case. 

```python
conv_cat = np.squeeze(conv_cat,axis=0)
conv_cat= conv_cat.transpose(2,0,1)
conv_cat.shape
```

**Output:**

After convolution we had 32 kernels going over the input image to learn different features of the image and that is why have 32,  (27X27) images. We see the dimensions have reduced from 28x28 original image to (27x27) after convolution. There are ways in Keras and PyTorch if we want to preserve the dimension of the original image. 

```
(32, 27, 27)
```

```python
fig= plt.figure(figsize=[25,8])
for i in range(32):
  ax= fig.add_subplot(4,8,i+1, xticks=[], yticks=[])
  plt.imshow(conv_cat[i])
```
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/full_cat.png" width="40%">
