[Google Colaboratory](https://colab.research.google.com/drive/18-qfthoIIMOv0RBEEGksfrp8hNM2wxau?usp=sharing)

## Loading the Image

1. Load the image
2. The default setting of the color mode in OpenCV comes in the order of BGR, which is different from that of Matplotlib. Therefore to see the image in GRAYSCALE mode, we need to convert it from BGR to GRAYSCALE as follows.
3. Convert it into a grayscale image


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc1.png" width="40%">

original image


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc2.png" width="40%">

grayscale image

## Blurring the Image

- The goal of blurring is to perform noise reduction.
- There are several techniques used to achieve blurring effects in OpenCV: Averaging blurring, Gaussian blurring, median blurring and bilateral filtering, **Non-Local Means Denoising**.
    - Types of Blurring


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc3.png" width="40%">

blurring the image 

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc4.png" width="40%">

de-noising the image 

## Thresholding

1. Thresholding transforms images into binary images. We need to set the threshold value and max values and then we convert the pixel values accordingly. 
2. There are five different types of thresholding: 
    - Binary,  cv2.THRESH_BINARY
    - the inverse of Binary,  cv2.THRESH_BINARY_INV
    - Threshold to zero,  cv2.THRESH_TOZERO
    - the inverse of Threshold to Zero,  cv2.THRESH_TOZERO_INV
    - and Threshold truncation.  cv2.THRESH_TRUNC
3. Adaptive thresholding, by calculating the threshold within the neighborhood area of the image, we can achieve a better result from images with varying illumination.
    - For Adaptive thresholding Image should be grayscale.
    - The parameters of adaptive thresholding are maxValue (255), adaptiveMethod , thresholdType , blockSize and C .
    - And the adaptive method here has two kinds: ADAPTIVE_THRESH_MEAN_C , ADAPTIVE_THRESH_GAUSSIAN_C


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc5.png" width="40%">


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc6.png" width="40%">

## **Edge Detection**

1. **[Edge detection](https://en.wikipedia.org/wiki/Edge_detection)** means identifying points in an image where the brightness changes sharply or discontinuously. We can draw line segments with those points, which are called ***edges***.
2. Then, we need to provide two values: threshold1 and threshold2. Any gradient value larger than threshold2 is considered to be an edge. Any value below threshold1 is considered not to be an edge.
3. Values in between threshold1 and threshold2 are either classiﬁed as edges or non-edges based on how their intensities are “connected”.

          canny = cv2.Canny(image, threshold1,  threshold2)


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc7.png" width="40%">

## Morphological transformations


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc8.png" width="40%">


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc9.png" width="40%">

## Contour detection

1. The cv2.findContours function stores a  numpy array of (x,y) points that from that contour

2. Retrieval Mode:

- cv2.CHAIN_APPROX_NONE – Stores all the points along the line
- cv2.CHAIN_APPROX_SIMPLE – Stores the end points of each line (efficient)

3. Sort the contours by Area to get the top 4 contours, i.e the edges in the image forming a square

4. Using cv2. approxpolyDP  function is used to approximate the contour, (with approximation accuracy, and closed (True or False) polygon). It will give the result of the number of individual lines or end points (4 in this case as it is square)

5. Using cv2.drawContours function draw the contour shape using the  identified contour which has length 4 in this example


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc10.png" width="40%">

## Affine Transformation

- Transformations are geometric distortions enacted upon an image
- It is used to correct distortions and perspective issues.
- Affine Transformation helps to modify the geometric structure of the image, preserving parallelism of lines but not the lengths and angles. It preserves collinearity and ratios of distances.


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc11.png" width="40%">

## Final Output

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc12.png" width="40%">

original image

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sc13.png" width="40%">

scanned image
