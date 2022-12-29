#!/usr/bin/env python3

import tempfile
import numpy as np
import pandas as pd
import os
import time
import yep
import urllib
import urllib.request

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from se.CShrubEnsembles import CDecisionTreeClassifier
# from se.CShrubEnsembles import CDecisionTreeClassifier, CDecisionTreeClassifierOpt

def download(url, filename, tmpdir = None):
    """Download the file under the given url and store it in the given tmpdir udner the given filename. If tmpdir is None, then `tempfile.gettmpdir()` will be used which is most likely /tmp on Linux systems.

    Args:
        url (str): The URL to the file which should be downloaded.
        filename (str): The name under which the downlaoded while should be stored.
        tmpdir (Str, optional): The directory in which the file should be stored. Defaults to None.

    Returns:
        str: Returns the full path under which the file is stored. 
    """
    if tmpdir is None:
        tmpdir = os.path.join(tempfile.gettempdir(), "data")

    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    if not os.path.exists(os.path.join(tmpdir,filename)):
        print("{} not found. Downloading.".format(os.path.join(tmpdir,filename)))
        urllib.request.urlretrieve(url, os.path.join(tmpdir,filename))
    return os.path.join(tmpdir,filename)

def benchmark(model, X_train, y_train, X_test, y_test, reps=5):
    fit_times = []
    for _ in range(reps):
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        fit_time = end - start
        fit_times.append(fit_time)

    predict_times = []
    for _ in range(500):
        start = time.time()
        preds = np.array(model.predict_proba(X_test))
        accuracy = accuracy_score(y_test, preds.argmax(axis=1))
        end = time.time()
        predict_time = end - start
        predict_times.append(predict_time)

    return {
        "fit_time":np.mean(fit_times),
        "predict_time":np.mean(predict_times),
        "test_accuracy":accuracy
    }

tmpdir = os.path.join(tempfile.gettempdir(), "data")

if not os.path.exists(os.path.join(tmpdir,"covtype.npy")):
    covtype_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", "covtype.data.gz", tmpdir)

    print("WARNING: Covertype not found. Preparing binary data format. This can take a lot of memory and should not be performed on resource-constraint devices")
    df = pd.read_csv(covtype_path, compression="gzip", header=None)
    df = df.dropna()
    df = df.sample(frac=1)
    np.save(os.path.join(tmpdir,"covtype.npy"), df.values)
    # df.values.tofile(os.path.join(tmpdir,"covtype.npy"))

data = np.load(os.path.join(tmpdir,"covtype.npy"))
X, Y = data[:,:-1], data[:,-1].astype(np.int32)
Y = Y - 1 # Make sure classes start with index 0 instead of index 1 

# NData = 1000 #[10000, 20000, 50000, 100000]
# X_train, X_test, y_train, y_test = X[0:NData,:], X[NData:2*NData,:], Y[0:NData], Y[NData:2*NData] 


max_depth = 0
n_classes = len(set(Y))
max_features = int(np.sqrt(X.shape[1]))
# print(max_features)
seed = 1234
step_size = 0
tree_init_mode = "train"
tree_update_mode = "none"
df = []

for N in [500, 1000, 5000]:
# for N in [1000, 10000, 100000, None]:
    if N is not None and 2 * N < X.shape[0]:
        X_train, X_test, y_train, y_test = X[0:N,:], X[N:2*N,:], Y[0:N], Y[N:2*N] 
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    print("Starting benchmark on {} training and {} testing".format(X_train.shape, X_test.shape))

    dt = CDecisionTreeClassifier(5,n_classes,max_features,seed,step_size,tree_init_mode,tree_update_mode)
    df.append({
        "name":"dt",
        "N":N,
        **benchmark(dt,X_train,y_train,X_test,y_test)
    })

    sktree = DecisionTreeClassifier(max_depth=5, max_features=max_features, splitter="best")
    df.append({
        "name":"sklearn",
        "N":N,
        **benchmark(sktree,X_train,y_train,X_test,y_test)
    })

df = pd.DataFrame(df)
print(df)
