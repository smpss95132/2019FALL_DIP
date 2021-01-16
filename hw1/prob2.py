#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# In[17]:


def Gaussian_noise(img, magnitude):
    noise = np.random.normal(0, magnitude, img.shape)
    result_image = img + noise
    result_image = result_image.astype('uint8')
    
    return result_image


# In[18]:


def Salt_and_pepper_noise(img, proportion):
    random_matrix = np.random.uniform(0,1,img.shape)
    result_image = np.zeros(img.shape, dtype='uint8')
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random_matrix[i][j] <= proportion:
                result_image[i][j] = 0
            elif random_matrix[i][j] >= (1-proportion):
                result_image[i][j] = 255
            else:
                result_image[i][j] = img[i][j]
                
    return result_image


# In[19]:


def low_pass_filter_average(noise_image, kernel_size):
    result_img = np.zeros(noise_image.shape, dtype=np.uint8)
    
    window_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(noise_image, window_size, window_size, window_size, window_size, cv2.BORDER_REFLECT) 
    
    for i in range(window_size, padded_image.shape[0]-window_size):
        for j in range(window_size, padded_image.shape[1]-window_size):
            window = padded_image[i-window_size:i+window_size+1,j-window_size:j+window_size+1]
            result_img[i-window_size][j-window_size] = int(np.average(window))
    
    return result_img   


# In[20]:


def median_filter(noise_image, kernel_size):
    result_img = np.zeros(noise_image.shape,dtype=np.uint8)
    
    window_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(noise_image, window_size, window_size,window_size, window_size, cv2.BORDER_REFLECT) 
    
    for i in range(window_size, padded_image.shape[0]-window_size):
        for j in range(window_size, padded_image.shape[1]-window_size):
            window = padded_image[i-window_size:i+window_size+1,j-window_size:j+window_size+1]
            result_img[i-window_size][j-window_size] = np.median(window)
    
    return result_img  


# In[21]:


def PSNR(img_1, img_2):
    MSE = np.sum((img_1 - img_2) ** 2) / (img_1.shape[0] * img_2.shape[1])
    
    return 10 * math.log(255.0**2/MSE, 10)


# In[22]:


path1 = "sample4.jpg"
img_sample4 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)


# In[23]:


# Problem 2(a)
img_resultG1 = Gaussian_noise(img_sample4, 10)
img_resultG2 = Gaussian_noise(img_sample4, 30)
cv2.imwrite("resultG1.jpg", img_resultG1)
cv2.imwrite("resultG2.jpg", img_resultG2)


# In[24]:


# Problem 2(b)
img_resultS1 = Salt_and_pepper_noise(img_sample4, 0.05)
img_resultS2 = Salt_and_pepper_noise(img_sample4, 0.15)

cv2.imwrite("resultS1.jpg", img_resultS1)
cv2.imwrite("resultS2.jpg", img_resultS2)


# In[25]:


# Problem 2(c)
noise_img_G1 = cv2.imread("resultG1.jpg", cv2.IMREAD_GRAYSCALE)
noise_img_G2 = cv2.imread("resultG2.jpg", cv2.IMREAD_GRAYSCALE)


# use average filter to deal with Gaussain noise
cleaned_img_G1 = low_pass_filter_average(noise_img_G1, 3)
cleaned_img_G2 = low_pass_filter_average(noise_img_G2, 9)

cv2.imwrite("resultR1.jpg", cleaned_img_G1)
cv2.imwrite("resultR2.jpg", cleaned_img_G2)


# In[26]:


# Problem 2(d)
noise_img_S1 = cv2.imread("resultS1.jpg", cv2.IMREAD_GRAYSCALE)
noise_img_S2 = cv2.imread("resultS2.jpg", cv2.IMREAD_GRAYSCALE)

cleaned_img_S1 = median_filter(noise_img_S1, 3)
cleaned_img_S2 = median_filter(noise_img_S2, 5)

cv2.imwrite("resultR3.jpg", cleaned_img_S1)
cv2.imwrite("resultR4.jpg", cleaned_img_S2)


# In[27]:


# Problem 2(e)
origin = cv2.imread("sample4.jpg", cv2.IMREAD_GRAYSCALE)
resultR1 = cv2.imread("resultR1.jpg", cv2.IMREAD_GRAYSCALE)
resultR2 = cv2.imread("resultR2.jpg", cv2.IMREAD_GRAYSCALE)
resultR3 = cv2.imread("resultR3.jpg", cv2.IMREAD_GRAYSCALE)
resultR4 = cv2.imread("resultR4.jpg", cv2.IMREAD_GRAYSCALE)

R1_PSNR = PSNR(origin, resultR1)
R2_PSNR = PSNR(origin, resultR2)
R3_PSNR = PSNR(origin, resultR3)
R4_PSNR = PSNR(origin, resultR4)

print(R1_PSNR)
print(R2_PSNR)
print(R3_PSNR)
print(R4_PSNR)

