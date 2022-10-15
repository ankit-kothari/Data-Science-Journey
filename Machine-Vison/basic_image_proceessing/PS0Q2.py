import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


#load the image
image = cv2.imread('inputPS0Q2.jpg')
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#figure with 3 rows and 2 columns subplots and the size of each subplot is 10 X 10
fig, ax = plt.subplots(3, 2, figsize=(10, 10))

#plot the image in the first subplot


image_GRB = image_RGB.copy()

#################################################################
#part 1
#################################################################
#swap the green and red channels
image_GRB[:,:,0] = image_RGB[:,:,1]
image_GRB[:,:,1] = image_RGB[:,:,0]
#save the image
cv2.imwrite('swapImgPS0Q2.png', image_GRB)
ax[0, 0].imshow(image_GRB)

print(f'Saving Image to swapImgPS0Q2.png')

#################################################################
#part 2
#################################################################
#plot the image in the second subplot
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#save the image
cv2.imwrite('grayImgPS0Q2.png', gray)
ax[0, 1].imshow(gray, cmap= 'gray')

print(f'Saving Image to grayImgPS0Q2.png')

#################################################################
#part 3a
#################################################################
#plot the image in the third subplot
image_negative = 255 - gray
#save the image
cv2.imwrite('negativeImgPS0Q2.png', image_negative)
ax[1, 0].imshow(image_negative, cmap='gray')

print(f'Saving Image to negativeImgPS0Q2.png')

#################################################################
#part 3b
#################################################################
#plot the image in the fourth subplot
image_flip = np.fliplr(gray)
#save the image
cv2.imwrite('flipImgPS0Q2.png', image_flip)
ax[1, 1].imshow(image_flip, cmap='gray')

print(f'Saving Image to flipImgPS0Q2.png')

#################################################################
#part 3c
#################################################################
#plot the image in the fifth subplot
image_avg = (gray + image_flip)/2
#save the image
cv2.imwrite('avgImgPS0Q2.png', image_avg)
ax[2, 0].imshow(image_avg, cmap='gray')

print(f'Saving Image to avgImgPS0Q2.png')

#################################################################
#part 3d
#################################################################
#plot the image in the sixth subplot
N = np.random.randint(0, 255, (533, 800))
#save the N matrix to a file in npy format
np.save('noise.npy', N)

#add noise to the gray image
gray_noise = gray + N

#clip the values to be between 0 and 255
gray_noise = np.clip(gray_noise, 0, 255)


#save the image
cv2.imwrite('addNoiseImgPS0Q2.png', gray_noise)
ax[2, 1].imshow(gray_noise, cmap='gray')

print(f'Saving Image to addNoiseImgPS0Q2.png')

#save the figure
plt.savefig('PS0Q2.png')

plt.show()