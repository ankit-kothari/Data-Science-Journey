## Introduction

This article goes over the work done in the linked [google colab notebook](https://colab.research.google.com/drive/1eV8oUuWg0nRHMxeUZtZZxXL_QE5HbrTb?usp=sharing) in a nutshell. The goal here is to explore powerful CNN and extend the model to an actual implementation using OpenCV to recognize handwritten digits on a piece of paper using CNN trained on a famous MNIST dataset. 

## Why use CNN vs ANN for Image Classificaton?

- ANN will generate over 100,000 parameters for even 28x28 images, which is not scalable for large real-world pictures, it will have millions of features, whereas in CNN's focuses on local connectivity and connected to a subset of neurons, and not all neurons will be connected to every other neuron.
- Secondly we will lose all the 2D information in ANN since we have to flatten the image vs CNN where we feed the 2D image directly.

              For example: a 28x28 image ,

              ANN will be flattened to 784 features as input

              CNN will use 28x28 image as input 

- Lastly, ANN works on similar images, centered images but not on images which differ in position, for example, a 28 x 28 image if 1 is at the center in all images ANN would work, but if in some images 1 is to the extreme right of the image, an ANN won't understand that relationship.

## MNIST DATA

```python
Training Data (60000, 28, 28)
Test Data (10000, 28, 28)
Train Labels (60000,)
Test Labels (10000,)
```

### Data Preprocessing

### Scaling Data

```python
x_train = x_train/255
x_test = x_test/255
```

### One Hot Encoding for the target Values

```python
y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)
```

### Reshaping the data

For a CNN, but we need to add one more dimension to show we're dealing with 1 RGB channel (since technically the images are in black and white, only showing values from 0-255 on a single channel), a color image would have 3 dimensions.

```python
Training Data (60000, 28, 28, 1)
Test Data (10000, 28, 28, 1)
```

## MODEL ARCHITECTURE

```python
model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER 
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

# https://keras.io/metrics/
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 
```

## MODEL ACCURACY

Model accuracy using the CNN turns out to be 98.76%

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/da72e826-470b-41f4-acd2-a997afdae734/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/da72e826-470b-41f4-acd2-a997afdae734/Untitled.png)

## LOADING IN THE SAMPLE IMAGE with HANDWRITTEN DIGIT's

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9082d3ed-ad7d-4a1a-b63a-d32afc5ee6e1/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9082d3ed-ad7d-4a1a-b63a-d32afc5ee6e1/Untitled.png)

## IMAGE PREPROCESSING

### Blurring and converting the image to grayscale

- The goal of blurring is to perform noise reduction. If we apply edge detection algorithms to the images with high resolution, we’ll get too many detected outcomes that we aren’t interested in.
    - Types of Blurring
- Complexity of gray level images is lower than that of color images. Features like brightness, contrast, edges, shape, contours, texture, perspective, shadows, and so on,  can be analyzed without addressing color. It is also computationally expensive
- Also, many functions in openCV expects the image to be in grayscale.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/55707be8-f529-4e65-84e6-f4abda6e6ed8/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/55707be8-f529-4e65-84e6-f4abda6e6ed8/Untitled.png)

### Applying Edge Detection

The most famous [edge detection](https://medium.com/sicara/opencv-edge-detection-tutorial-7c3303f10788) method is the ***Canny Filter.* The Canny filter thresholds can be tuned to catch only the strongest edges and get cleaner contours. The higher the thresholds, the cleaner the edges.**

- Concept behind canny filtering

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27229669-858e-43c5-b706-2f0db2adbb66/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27229669-858e-43c5-b706-2f0db2adbb66/Untitled.png)

### Using OpenCV's dilation to enhance the image

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7b4c24ef-e708-4a89-898a-309bfc143b51/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7b4c24ef-e708-4a89-898a-309bfc143b51/Untitled.png)

### Contour Detection

[Contours](https://towardsdatascience.com/computer-vision-for-beginners-part-4-64a8d9856208), on the other hand, are closed curves which are obtained from edges and depicting a boundary of figures. In here the opeCV's cv2.findcontours will find each of these numbers (using certain conditions and manipulations)  and will be extracted and fed into the model to make predictions. Before applying the detection algorithm, we need to convert the image into grayscale and apply thresholding like we have done in the previous steps. 

## MAKING PREDICTIONS

```python
for rect in rects:
 if rect[2]>5 and rect[3]>=25:
  #cv2.rectangle(blurred, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
  # Make the rectangular region around the digit
  leng = int(rect[3] * 1.3)
  pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
  pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
  roi = dilation[pt1:pt1+leng, pt2:pt2+leng]

  # Resize the image
  roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
  #roi = cv2.dilate(roi, (3, 3))
  plt.imshow(roi,cmap='gray')
  plt.show()
  roi=roi/255
  roi=roi.reshape(1,28,28,1)
  
  preds1= auto_encoder.predict(roi)
  preds_name1 = np.argmax(preds1, axis=1)
  print(preds_name1)
  number = str(int(float(preds_name1)))
  cv2.putText(blurred, number, (rect[0], rect[1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
  plt.imshow(blurred,cmap='gray')
```

This part extracts these images from the original picture, and reshapes it and feeds into the model to generate predictions and annotate the image with the predicted label

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51c69634-4b44-4c59-a0be-1b3d9432ffe0/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51c69634-4b44-4c59-a0be-1b3d9432ffe0/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/838dca58-ff9a-419e-9d8f-7e701d8192c7/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/838dca58-ff9a-419e-9d8f-7e701d8192c7/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5cc1d961-0f38-4367-bd84-da7fbc137c05/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5cc1d961-0f38-4367-bd84-da7fbc137c05/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/47fc4327-0f92-422c-ac27-c213fb712505/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/47fc4327-0f92-422c-ac27-c213fb712505/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6262738e-8207-4339-bc49-35d21eae93fb/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6262738e-8207-4339-bc49-35d21eae93fb/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/59b8388d-6c5d-4931-9576-6426079ccaa8/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/59b8388d-6c5d-4931-9576-6426079ccaa8/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/826071c9-e0c2-4b0b-a596-396883646871/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/826071c9-e0c2-4b0b-a596-396883646871/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2d44501-3eff-4e5e-aab6-32b24e09a137/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2d44501-3eff-4e5e-aab6-32b24e09a137/Untitled.png)

## Results

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b772d344-2d27-4ebb-b5c2-c5960a15d7a5/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b772d344-2d27-4ebb-b5c2-c5960a15d7a5/Untitled.png)

Resources:

- [https://towardsdatascience.com/@jiwon.jeong](https://towardsdatascience.com/@jiwon.jeong)
- [https://www.geeksforgeeks.org/python-bilateral-filtering/](https://www.geeksforgeeks.org/python-bilateral-filtering/)
