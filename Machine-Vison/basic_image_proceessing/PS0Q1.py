import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2





#create a 100 X 100 matrix
x = np.random.randint(0, 255, (100, 100))

#save the matrix to a file in npy format
np.save('inputAPS0Q1.npy', x)

#load the matrix from the file
A = np.load('inputAPS0Q1.npy')

#plot the intensity value in descending order

plt.plot(np.sort(A.flatten())[::-1])



######################################################
#part 4a
######################################################

#plot the intensity value in descending order

plt.plot(np.sort(A.flatten())[::-1])

#LABEL THE AXES
plt.xlabel('PIXEL AFTER SORTING')
plt.ylabel('INTENSITY VALUE')
#TITLE THE PLOT
plt.title('INTENSITY VALUE IN DESCENDING ORDER')
plt.show()


######################################################
#part 4b
######################################################

#plot the distribution of intensity values

plt.hist(A.flatten(), bins=20, range=(0, 255), color='c', edgecolor='k', linewidth=1.0, rwidth=0.9)

#LABEL THE AXES
plt.xlabel('INTENSITY VALUE')
plt.ylabel('NUMBER OF PIXELS')
#TITLE THE PLOT
plt.title('DISTRIBUTION OF INTENSITY VALUES')
plt.show()

######################################################
#part 4
######################################################

# create bottom left quadrant of A
X = A[50:, 0:50]
X.shape

#plot X as an image
plt.imshow(X)
plt.show()

#save X  in npy format
np.save('outputXPS0Q1.npy', X)

######################################################
#part 4d
######################################################

#Subtract mean of A from each element of A

Y = A - np.mean(A)

#plot Y as an image
plt.imshow(Y)
plt.show()

#save Y  in npy format
np.save('outputYPS0Q1.npy', Y)


######################################################
#part 4e
######################################################




Z = np.zeros((100,100, 3))  # create a 100 X 100 X 3 matrix
t = np.mean(A)

# set the color channel to only red where the intensity is greater than the mean intensity of the image in every channel


Z[A > t] = [1, 0, 0]
# Else set z to black
Z[A <= t] = [0, 0, 0]

plt.imshow(Z)

#plot Z as an image
plt.show()          

#save Z  image in png format
plt.imsave('outputZPS0Q1.png', Z)