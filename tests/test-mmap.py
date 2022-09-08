#!/usr/bin/env python3

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from se.GASE import GASE
from se.MASE import MASE
from se.OSE import OSE

import os, psutil
process = psutil.Process(os.getpid())

# df = pd.read_csv("HIGGS.csv.gz", compression="gzip", header=None)
# df.values.tofile("higgs.bin")
#data = np.memmap('/home/sbuschjaeger/Downloads/higgs.bin', mode='r', dtype="float64").reshape( (-1,29) )
data = np.load('/home/sbuschjaeger/Downloads/higgs.npy', mmap_mode="r")

X,Y = data[:,1:], data[:,0].astype(np.int32)

NTotal = X.shape[0]
NTest = int(0.33 * NTotal)
NTrain = NTotal - NTest

print("Memory after loading {}".format(process.memory_info().rss / 10**6))
print("Starting split")
XTrain, XTest = X[0:NTrain], X[NTrain:]
YTrain, YTest = Y[0:NTrain], Y[NTrain]

print("Memory after split {}".format(process.memory_info().rss / 10**6))

# model = MASE(
#     max_depth = 15,
#     seed = 12345,
#     burnin_steps = 50,
#     max_features = 0,
#     loss = "mse",
#     step_size = 1e-1,
#     optimizer = "adam", 
#     tree_init_mode = "train", 
#     n_trees = 32, 
#     n_worker = 4,
#     n_rounds = 500,
#     batch_size_per_worker = 1024, 
#     bootstrap = True,
#     verbose = True,
#     out_path = None,
#     sample_engine="python",
#     batch_size=2**15
# )

# model = GASE(
#     max_depth = 15,
#     seed = 12345,
#     max_features = 0,
#     loss = "mse",
#     step_size = 1e-1,
#     optimizer = "adam", 
#     tree_init_mode = "train", 
#     n_trees = 16, 
#     n_rounds = 500,
#     n_workers = 4,
#     bootstrap = True,
#     verbose = True,
#     out_path = None,
#     sample_engine="python",
#     batch_size=2**12
# )

model = OSE(
    max_depth = 15,
    seed = 12345,
    burnin_steps = 0,
    max_features = 0,
    loss = "mse",
    step_size = 1e-2,
    optimizer = "adam", 
    tree_init_mode = "train", 
    batch_size = 2**10, 
    epochs=5,
    regularizer="hard-L0",
    # regularizer=None,
    l_reg=32,
    bootstrap = True,
    verbose = True,
    out_path = None
)

model.fit(XTrain,YTrain)
