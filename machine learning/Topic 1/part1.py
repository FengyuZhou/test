
# coding: utf-8

# In[1]:

from numpy import genfromtxt
my_data = genfromtxt('bv_data.csv', delimiter=',')
n = len(my_data)
x = my_data[1:n,0]
y = my_data[1:n,1]


# In[2]:

from sklearn import cross_validation
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import pylab
n_folds = 5
for d in [1,2,6,12]:
    err_train = []
    err_validation = []
    for N in range(20,105,5):
        kf = cross_validation.KFold(N, n_folds)
        e_tr = 0
        e_te = 0
        for train_index,validation_index in kf:
            x_train = x[train_index]
            y_train = y[train_index]
            x_validation = x[validation_index]
            y_validation = y[validation_index]
            coef = np.polyfit(x_train, y_train, d)
            e_tr += linalg.norm(np.polyval(coef, x_train)-y_train)**2/len(train_index)
            e_te += linalg.norm(np.polyval(coef, x_validation)-y_validation)**2/len(validation_index)
        err_train.append(e_tr/n_folds)
        err_validation.append(e_te/n_folds)
    plt.figure();
    plt.plot(range(20,105,5), err_train, label = 'Train error')
    plt.plot(range(20,105,5), err_validation, label = 'Validation error')
    pylab.legend(loc='upper right')
    plt.title('d='+str(d))
    plt.xlabel('N')
    plt.ylabel('error')
    plt.savefig('d='+str(d)+'.eps', format='eps', dpi=1000)
    plt.show()


# In[ ]:



