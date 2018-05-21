#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:31:53 2018

@author: Young
"""


# Import modules
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Import custom functions
utilities_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/NN/Functions'
nn_class_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/NN/Class'
sys.path.append(utilities_dir)
sys.path.append(nn_class_dir)
from nn import nn
from eval_util import mse
from munge_util import binarize_labels
from parallel_util import get_num_cores
from nn_functions import logistic
from nn_functions import softmax

# Import parallelisation modules
import multiprocessing
from joblib import Parallel, delayed


# =============================================================================
# Main
# =============================================================================
# Import some data to play with
#iris = datasets.load_iris()
#data = pd.DataFrame(iris.data)
#data.columns = iris.feature_names
#labels = pd.DataFrame(iris.target)
#labels.columns = ['labels']

data, labels = datasets.make_moons(100000, noise=0.1)
data = pd.DataFrame(data)
labels = pd.DataFrame(labels)
labels.columns = ['labels']

#plt.scatter(data.iloc[:,0], data.iloc[:,1], c=list(map(lambda x: ['red', 'blue'][int(x)], np.array(labels))))
#plt.show()

# Normalise data
normalised_data = (data - np.mean(data))/np.std(np.array(data),0)

# Split data
train_x,test_x,train_y,test_y = train_test_split(normalised_data, labels, train_size = 0.6, test_size = 0.4, random_state=1)
bin_train_y = train_y
bin_test_y = test_y

if __name__ == '__main__':
    # Initialise
    nnet = nn(layers = [2,4,1], 
              sigmoids = [logistic, logistic], 
              random_seed = 12)
    
    # Train
    ncores = get_num_cores()
    nn_params = {'learning_rate': 1,
                 'influence_of_inertia': 0.1,
                 'size_minibatch': 60000, 
                 'epochs': 10,
                 'error_func': mse,
                 'verbose': True,
                 'n_jobs_data_parallelisation': 8}
    nnet.fit(x=train_x, y=bin_train_y, **nn_params)
    
    # Validate
#    np.round(nnet.predict(x=test_x),2)
    nnet.validate(x=test_x, y=bin_test_y, verbose=True)
#
## Save
#nn_save_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/NN/Models'
#nn_save_name = 'nn_10k.pkl'
#nnet.save(os.path.join(nn_save_dir, nn_save_name))
#
## Load saved net
#nnet2 = nn.load('/Users/Young/Documents/Capgemini/Learning/Machine Learning/NN/Models/nn_1000.pkl')
#nnet2.validate(x=test_x, y=bin_test_y)
#
#pred_classes = np.argmax(nnet.predict(x=test_x),1)
#true_classes = np.argmax(np.array(bin_test_y),1)
#plot_conf_mat(true_classes, pred_classes, figsize=(5,5))
#
#
#
#pd.DataFrame(true_classes == pred_classes)
#preds = nnet.predict(x=test_x)
#y = bin_test_y
#
#
#
#14.45 +8.95 + 15.45 -15 - 17.5