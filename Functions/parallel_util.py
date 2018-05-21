#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:57:09 2018

@author: Young
"""

# Import parallelisation modules
import random
import multiprocessing
from joblib import Parallel, delayed

def get_num_cores():
    return multiprocessing.cpu_count()

def chunk_list(your_list, chunk_count):
    # Chunk size
    chunk_size = int(len(your_list) / float(chunk_count))
    
    # Shuffle indices
    random_indices = random.sample(your_list, k=len(your_list))
    
    # Get chunks
    chunks = []
    for chunk_num in range(chunk_count):
        idx_start = chunk_num * chunk_size
        idx_end = (chunk_num+1) * chunk_size
        chunks.append(random_indices[idx_start:idx_end])
    
    return chunks
#def parallelise(map_job, ncores, ):
#    Parallel(n_jobs = ncores)(delayed(map_job)() for path in list)