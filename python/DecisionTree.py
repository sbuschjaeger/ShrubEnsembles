import numpy as np
import random
from scipy.special import softmax
from sklearn.utils.multiclass import unique_labels
import os
from tqdm import tqdm
import gzip
import pickle

import ShrubEnsembles
from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin

class DecisionTree(ClassifierMixin, BaseEstimator):

    def __init__(self,
        max_depth = 1, 
        max_features = 0, 
        random_state = 0, 
        step_size = 0,
        tree_init_mode = "train"
    ):

        tree_init_mode = tree_init_mode.lower()
        assert tree_init_mode in ["train", "random"], f"Currently only the tree_init_modes {{train, random}} are supported but you supplied {tree_init_mode}"

        if step_size < 0:
            print(f"WARNING: You supplied a negative step size of {step_size}. Do you want to de-optimize?")
        
        # Use the more common "random_state" name instead of seed, although the c++ backend uses seed
        if random_state is None:
            self.seed = 1234
        else:
            self.seed = random_state

        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.max_depth = max_depth
        self.max_features = max_features
        self.step_size = step_size
        self.tree_init_mode = tree_init_mode

        self.model = None

    def predict_proba(self, X):
        ''' Predict class probabilities using the pruned model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,C)
            The predicted class probabilities. 
        '''
        assert self.model is not None, "Call fit before calling predict_proba!"

        if len(X.shape) < 2:
            # The c++ bindings only supports batched data and thus we add the implicit batch dimension via data[np.newaxis,:]
            return np.array(self.model.predict_proba(X[np.newaxis,:]))[0]
        else:
            return np.array(self.model.predict_proba(X))

    def predict(self, X):
        ''' Predict classes using the pruned model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted classes. 

        '''
        proba = self.predict_proba(X)
        return self.classes_.take(proba.argmax(axis=1), axis=0)

    def num_bytes(self):
        assert self.model is not None, "Call fit before calling num_trees!"
        return self.model.num_bytes()

    def num_nodes(self):
        assert self.model is not None, "Call fit before calling num_nodes!"
        return self.model.num_nodes()

    def load(self, nodes):
        assert self.model is not None, "Call fit before calling num_nodes!"
        self.model.load(nodes)

    def store(self):
        assert self.model is not None, "Call fit before calling num_nodes!"
        return self.model.nodes()

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        if self.tree_init_mode == "train":
            from ShrubEnsembles.ShrubEnsembles import DecisionTree_DOUBLE_TRAIN
            self.model = DecisionTree_DOUBLE_TRAIN(self.n_classes_, self.max_depth, self.max_features, self.seed, self.step_size)
        else:
            from ShrubEnsembles.ShrubEnsembles import DecisionTree_DOUBLE_RANDOM
            self.model = DecisionTree_DOUBLE_RANDOM(self.n_classes_, self.max_depth, self.max_features, self.seed, self.step_size)
        
        self.model.fit(X,y)