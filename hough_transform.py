#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
import math
import cv2


# # Hough transform function

# In[2]:


def hough_line(edge):
    # Theta 0 - 180 degree
    # Calculate 'cos' and 'sin' value ahead to improve running time
    theta = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))

    # Generate a accumulator matrix to store the values
    rho_range = round(math.sqrt(edge.shape[0]**2 + edge.shape[1]**2))
    accumulator = np.zeros((rho_range, len(theta)), dtype=np.uint8)

    # Threshold to get edges pixel location (x,y)
    edge_pixels = np.where(edge == 255)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    # Calculate rho value for each edge location (x,y) with all the theta range
    for p in range(len(coordinates)):
        for t in range(len(theta)):
            rho = int(round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]))
            accumulator[rho, t] += 1 # Suppose add 1 only, Just want to get clear result

    return accumulator


# # Main function

# In[35]:


def detect_lines(image_name):
    image = cv2.imread(image_name)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thres_1 = 100
    thres_2 = 200
    edges = cv2.Canny(grayscale,thres_1,thres_2)

    # Function to do hough line transform
    accumulator = hough_line(edges)

    threshold  = 60
    x,y = accumulator.shape
    acc_values = np.array([np.amax(accumulator)])
    for i in range(0,x):
        for j in range(0,y):
            if accumulator[i,j] > threshold :
                acc_values = np.append(acc_values,accumulator[i,j])
    # Threshold some high values then draw the line
    edge_pixels = np.where(accumulator > threshold)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    # Use line equation to draw detected line on an original image
    for i in range(0, len(coordinates)):
        a = np.cos(np.deg2rad(coordinates[i][1]))
        b = np.sin(np.deg2rad(coordinates[i][1]))
        x0 = a*coordinates[i][0]
        y0 = b*coordinates[i][0]
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)

    # show result
    cv2.imshow('edges',edges)
    cv2.imshow('image', image)
    cv2.imshow('accumulator',accumulator)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    shortest_line = np.amin(acc_values)
    longest_line = np.amax(acc_values)
    print(' Total number of lines : ',len(acc_values)+1)
    print('Length of shortest line : ',shortest_line)
    print('Length of longest line : ',longest_line)
    return 


# In[36]:


image_name = 'aa016560.jpg'     
detect_lines(image_name)


# # Using opencv built-in function

# In[4]:


import cv2
import numpy as np

img = cv2.imread('5_sgw9.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 400, 800, apertureSize=3)
imS_edges = cv2.resize(edges, (960, 540))
cv2.imshow('edges', imS_edges)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
    x1 = int(x0 + 10000 * (-b))
    # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
    y1 = int(y0 + 10000 * (a))
    # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
    x2 = int(x0 - 10000 * (-b))
    # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
    y2 = int(y0 - 10000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

imS = cv2.resize(img, (960, 540))
cv2.imshow('image', imS)
k = cv2.waitKey(0)
cv2.destroyAllWindows()


# In[132]:





# In[ ]:





# In[ ]:




