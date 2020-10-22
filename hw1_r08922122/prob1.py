#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# In[37]:


# Problem 1(a)
def Histogram(img):
    
    gray_scale_count = [0] * 256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray_scale_count[img[i][j]] += 1
            
    index = np.arange(256)
    plt.bar(index,gray_scale_count)
    plt.grid(True)
    plt.show()
    
    return gray_scale_count


# In[38]:


# Problem 1(b)
def Global_histogram_equalization(img):    
    gray_scale_count = [0] * 256
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray_scale_count[img[i][j]] += 1
    
    total_pixel = img.shape[0]*img.shape[1]
    accumulator = 0
    look_up_table = [0] * 256
    for i in range(256):
        accumulator += gray_scale_count[i]
        look_up_table[i] = (accumulator/total_pixel) * 255
    
    
    result_image = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #print(look_up_table)
            result_image[i][j] = look_up_table[img[i][j]]
            
    Histogram(result_image)

    return result_image


# In[39]:


# Problem 1(c)
def Local_histogram_equalization(img, window_size):
    
    neightbor_length = (window_size-1) // 2
    
    padded_img = np.zeros((img.shape[0]+2*neightbor_length,img.shape[1]+2*neightbor_length), dtype = 'uint8')
    padded_img[neightbor_length:neightbor_length+img.shape[0], neightbor_length:neightbor_length+img.shape[1]] = img

    result_image = np.zeros(img.shape, dtype='uint8')
    total_pixels = window_size * window_size
    for i in range(neightbor_length, neightbor_length+img.shape[0]):
        for j in range(neightbor_length, neightbor_length+img.shape[1]):
            
            compare_result = (padded_img[i-neightbor_length:i+neightbor_length+1,j-neightbor_length:j+neightbor_length+1] <= padded_img[i][j])
            #print(compare_result)
            rank = np.sum(compare_result)
            #region_max = np.max(padded_img[i-neightbor_length:i+neightbor_length+1,j-neightbor_length:j+neightbor_length+1])
            #region_min = np.min(padded_img[i-neightbor_length:i+neightbor_length+1,j-neightbor_length:j+neightbor_length+1])

            #result_image[i-neightbor_length][j-neightbor_length] =  (rank / total_pixels) * (region_max-region_min) + region_min
            result_image[i-neightbor_length][j-neightbor_length] = (rank / total_pixels) * 255
            #print(i, j, rank, region_min, region_max)
    result_image = result_image.astype('uint8')
    
    return result_image             


# In[40]:


# Problem 1(e)
def log_transform(img, a):
    result_img = np.zeros(shape=img.shape)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result_img[i][j] = math.log(1+a*(img[i][j]/255.0)) / math.log(2)
            
    result_img = (result_img * 255.0).astype('uint8')
    
    return result_img
            
def inverse_log_transform(img):
    result_img = np.ones(shape=img.shape)   
    result_img = result_img - (img/255.0)
    result_img = (result_img * 255.0).astype('uint8')
    
    return result_img

def power_law_transform(img, p):
    result_img = np.zeros(shape=img.shape)
    result_img = (img/255.0) ** p
    result_img = (result_img * 255.0).astype('uint8')
    
    return result_img


# In[41]:


# load image
path1 = 'result1.jpg'
img_result1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
path2 = 'sample3.jpg'
img_sample3 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)


# In[42]:


# Problem 1(a)
result_image_1 = Histogram(img_result1)
result_image_2 = Histogram(img_sample3)


# In[43]:


# Problem 1(b)
result_image = Global_histogram_equalization(img_sample3)
cv2.imwrite('result5.jpg', result_image)


# In[44]:


# Problem 1(c)
result_image = Local_histogram_equalization(img_sample3 , 9)
Histogram(result_image)
cv2.imwrite('result6.jpg', result_image)


# In[45]:


# Problem 1(e)

img_result7 = log_transform(img_sample3,3)
img_result8 = inverse_log_transform(img_result7)
img_result9 = power_law_transform(img_sample3, 0.5)

Histogram(img_sample3)
Histogram(img_result7)
Histogram(img_result8)
Histogram(img_result9)

cv2.imwrite('result7.jpg', img_result7)
cv2.imwrite('result8.jpg', img_result8)
cv2.imwrite('result9.jpg', img_result9)

