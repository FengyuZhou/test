
# coding: utf-8

# In[1]:

from numpy import genfromtxt
import numpy as np
from numpy import linalg
import random
my_data = genfromtxt('sgd_data.csv', delimiter=',')
n = len(my_data)
x = my_data[1:n,0:4]
y = my_data[1:n,4]
x = np.hstack((np.ones((n-1,1)),x))
w_opt = (np.mat(x).T*np.mat(x)).I*(np.mat(x).T*np.mat(y).T)


# In[2]:

eta = np.exp(-15)
w = np.mat(0.001*np.ones(5))
N_epoch = 1000
epsilon = 0.0001
d01 = 0
err_start = linalg.norm(y-w*np.mat(x).T)**2
err_end = err_start
index = range(0,n-1)
while d01==0 or (err_start-err_end) > epsilon*d01:
    err_start = err_end
    random.shuffle(index)
    x = x[index,:]
    y = y[index]

    for it in range(0,N_epoch):
        dw = 2*(y[it]-w*np.mat(x[it,:]).T)*x[it,:]
        w += eta * dw

    err_end = linalg.norm(y-w*np.mat(x).T)**2
    if(d01==0):
        d01 = err_start-err_end


# In[3]:

print w
print w_opt.T


# In[ ]:



