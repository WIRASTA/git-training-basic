#!/usr/bin/env python
# coding: utf-8

# In[1]:


# time module for calculating execution time
import time


# In[2]:


# Counter method
from collections import Counter


# In[3]:


xrange=range


# In[4]:


# list to store perimeters
perimeters = []

# looping to generate a,b and c
for a in xrange(1, 500):
    for b in xrange(a, 500):
        c = (a**2 + b**2)**0.5
        if int(c) == c and a + b + c <= 1000:
            perimeters.append(a+b+c)


# In[5]:


# counting the instances of perimeters
p = Counter(perimeters)

# printing the most common perimeter
# or printing the most repeated perimeter
print (p.most_common(1)[0])

