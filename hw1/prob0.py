#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[10]:


# Problem 0(a)
def Raw_to_jpg(path):
    imgData = np.fromfile(path, dtype = 'uint8')
    imgData = imgData.reshape(400,600,1)
    
    cv2.imwrite('result1.jpg', imgData)


# In[11]:


# Problem 0(b)
def color2gray_scale(img):
    result_img = img[:,:,0]*0.07 + img[:,:,1]*0.72 + img[:,:,2]*0.21
    result_img = result_img.astype('uint8')

    return result_img


# In[12]:


# Problem 0(c)
def Rotate90(img):
    result_img = np.rot90(img)
    return result_img
    
def Diagonal_flip(img):
    result_img = np.transpose(img)
    return result_img


# In[13]:


# Problem 0(a)
Raw_to_jpg('sample1.raw')


# In[14]:


# Problem 0(b)
path = "sample2.jpg"
img = cv2.imread(path, cv2.IMREAD_COLOR)
result_img = color2gray_scale(img)
cv2.imwrite('result2.jpg', result_img)


# In[15]:


# Problem 0(c)
path = "result2.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
result_image_1 = Rotate90(img)
result_image_2 = Diagonal_flip(img)
cv2.imwrite('result3.jpg', result_image_1)
cv2.imwrite('result4.jpg', result_image_2)

