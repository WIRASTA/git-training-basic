#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import time module for calculating execution time
import time


# In[2]:


# largest pandigital number
largest = 0


# In[3]:


xrange=range


# In[4]:


# for loop to loop till 4 digits
for i in xrange(1,10000):

    # value for concatenated product
    multiplication = ''
    
    # (1,2,3,4,.....n)
    integer = 1

    # if the multiplication < 9 digits
    while len(multiplication) < 9:
        
        # concatenating the product at each stage
        multiplication += str(i*integer)
        
        # incrementing (1,2,3,4,....n)
        integer += 1
        
    # check for digits less than 9
    # check for all 1-9 numbers
    # check if '0' not in concatenated sting
    if ((len(multiplication) == 9) and (len(set(multiplication)) == 9) 
        and ('0' not in multiplication)):
    
        # check if multiplication is greater than largest
        if int(multiplication) > largest:
            largest = int(multiplication)


# In[5]:


# printing the largest
print (largest)

