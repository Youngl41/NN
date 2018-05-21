#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 21:50:24 2018

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
from math import sqrt
# Import parallelisation modules
import multiprocessing
from joblib import Parallel, delayed

from datetime import datetime
def hi(row):
    return (row[0] + row[1]) ** 2 / 0.14
#
#    
#if __name__ == '__main__':
#    data, labels = datasets.make_moons(1000000, noise=0.1)
#    st = datetime.now()
#    r = Parallel(n_jobs = 8)(delayed(hi)(i) for i in data)
#    print(datetime.now() - st)
    
    
if __name__ == '__main__':
    #data, labels = datasets.make_moons(500000, noise=0.1)
    st = datetime.now()
    # Parallel job - extract features
    ncores         = multiprocessing.cpu_count()
    a = Parallel(verbose = 1, n_jobs=ncores)(delayed(sqrt)(i**2) for i in range(1000000))
    #r = Parallel(verbose=1, n_jobs = 2)(delayed(hi)(i) for i in data)
    print(datetime.now() - st)
#    
#if __name__ == '__main__':
#    st = datetime.now()
#    a = []
#    for i in data:
#        a.append(hi(i))
#    print(datetime.now() - st)