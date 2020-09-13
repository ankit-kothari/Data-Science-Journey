## pytorch Implementation

[Google Colaboratory](https://colab.research.google.com/drive/1FPcPOsv0Vp9g_l4MXj3sag8UfXacslhk?usp=sharing)

## What is Style Transfer?

- For  style, the earlier layers provide for a more "localized" representation. This is opposed to the content model, where the later layers represent a more "global" structure.

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/st1.jpeg" width="80%">

## Creating  the VGG model function

1. VGG19 features portion is the CNN layers
2. keeping requires grad = False keeps the parameters unaffected during gradient decent and backpropogation
3. torch.device function to use GPU
4. connect the model to the device

## Image preprocessing function

1. Unsqueeze adds an extra dimension (just like np.expand.dims) (batch size, H, W, channel)
2. Image is resized to the maximum (400 pixel this case) dimension
3. Image is normalized to mean=0.5 and stdv= 0.5
4. Image is converted to a tensor

## Loading the content and the style image

1. Convert the image into 'RGB' format
2. Load the images and converted into TENSOR
3. Pass in the shape parameter so as the two images are of the same size

## Convert the images into numpy array for Visualization

1. Tensor passed in is in shape (batch_size, color channel, H, W)
2. Clone before to converting to numpy array
3. then squeeze the batch_size dimension
4. Transpose the array in shape (H,W, color channel) for matplotlib
5. denormalize the image
6. Clip the image so that it is in 0 to 1 range

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/st1.jpeg" width="40%">

## Get Features for the intermediate layers for content and style layer outputs (Targets)

1. Initializing the layers that will be used for content and style output
2. Create a dictionary to store the output of the image at each layer
3. Create a feature dictionary for content image
4. Create a feature dictionary for style image

## Define function for Gram Matrix for  style loss

1. Create a gram matrix function
2. Loop in each output of the style features and create a gram matrix dictionary.

$$style loss = mean((gram_matrix(y) - grammatrix(t))**2)$$

Gram Matrix : 

- Input Shape: (H,W,C)
- It’s then converted into (H*W, C) named as X
- Multiply with its transpose.
- Resulting Shape: (C,C)
- This makes the network loses the spatial information and only keep the style features of the style image

X^TX/N

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/st2.jpeg" width="80%">


## Assigning weight to style and content layer loss

1. In style features earlier layers are more important than the later layers opposite of the content extraction
2. Ratio is set for each of the style layer outputs
3. also alpha/beta is the ratio of content_weight/style weigh can have a big affect on the final image transfer

## Initializing the Input Image (Inputs)

1. Initializing the Input Image
2. This can be random noise for an image but we will be taking the clone of the content image
3. Using the requires_grad_(True) parameter to optimize the input image wrt to style and content images.

## Declaring the *"Adam"* optimizer to the Input image.

1. show_every = 300, to show our style transfer progress at every 300 steps.
2. Initializing the optimizer to optimize the input image
3. Run the optimization for 300 steps to get decent results

## Optimization and loss calculation

1. Create a loop to optimize for "steps" count.
2. calculate the features using the input image and VGG model.
3. calculate the content loss using MSE.

    $$Content Loss =  mean((target - Input)**2)$$

    - Here the target is the predicted value from the content image with pre-trained weights and bias, (like the one image where to want to copy the content from eg image of a cricketer MS Dhoni here.)
    - Input is the randomly initialized image  (starting with randomly initialized value and trying to minimize the loss with respect to the content image (target) so that we capture/copy good features from the content image to the randomly initialized image.
4. calculate the style loss
    - Calculate the gram matrix for input image and the target image at each of the convolution (the one which is randomly initialized and the style image)
    - At each step calculate the difference between the two and MSE on top of it calling the style loss function and sum it up at the end of all conv blocks

    $$Style  Loss = mean((gram_matrix(y) - grammatrix(t))**2)$$

5. Calculate total loss and apply the weights 

$$totalloss = contentloss + styleloss *styleweight$$

6.  Update the gradients with respect to the input image

- These input image is optimized  to make it  similar to the content image at the chosen convolution (5th out of 16 here in this example, can be anything) and style is optimized from the style image using the loss optimization at each of the chosen convolution.

Here Loss is calculated after each convolution using the Style Loss function

The gradients are calculated, loss with respect to input image and not the parameters like (W and b in neural networks).

## Final Output

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/st_final.jpeg" width="40%">

