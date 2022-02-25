import numpy as np
import random
from scipy.special import softmax
from sklearn.utils.multiclass import unique_labels
import time
import os
from tqdm import tqdm
import gzip
import pickle

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin

from se.CShrubEnsembles import COSE


# Modified from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
def create_mini_batches(inputs, targets, batch_size, shuffle=False, with_replacement=False):
    assert inputs.shape[0] == targets.shape[0]
    if with_replacement:
        assert shuffle == True

    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    start_idx = 0
    while start_idx < len(indices):
        if not with_replacement:
            if start_idx + batch_size > len(indices) - 1:
                excerpt = indices[start_idx:]
            else:
                excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = np.random.choice(indices, size = batch_size, replace=with_replacement)

        start_idx += batch_size

        yield inputs[excerpt], targets[excerpt]

# TODO Add Regressor
class OSE(ClassifierMixin, BaseEstimator):

    def __init__(self,
        max_depth = 5,
        seed = 12345,
        normalize_weights = True,
        burnin_steps = 0,
        max_features = 0,
        loss = "mse",
        step_size = 1e-2,
        optimizer = "sgd", 
        tree_init_mode = "train", 
        regularizer = "none",
        l_reg = 0,
        batch_size = 32, 
        epochs = 5,
        verbose = False,
        out_path = None,
        bootstrap = False
    ):

        # TODO set n_jobs for openmp?

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert regularizer is None or regularizer in ["none","L0", "L1", "hard-L0"], "Currently only {{none,L0, L1, hard-L0}} the regularizer regularizer is supported"
        assert l_reg >= 0, "l_reg must be greater or equal to 0"

        assert max_depth >= 0, "max_depth must be >= 0. Use max_depth = 0 if you want to train unlimited trees. You supplied: {}".format(max_depth)

        assert max_features >= 0, "max_features must be >= 0. Use max_features = 0 if you want to evaluated all splits. You supplied: {}".format(max_features)

        assert epochs > 0, "Epochs should be at least 1, but you gave {}".format(epochs)

        if batch_size is None or batch_size < 1:
            print("WARNING: batch_size should be greater than 2 for ShrubEnsemble for optimal performance, but was {}. Fixing it for you.".format(batch_size))
            batch_size = 2

        if regularizer == "hard-L0" and l_reg < 1:
            print("WARNING: You set l_reg to {}, but regularizer is hard-L0. In this mode, l_reg should be an integer 1 <= l_reg <= max_trees where max_trees is the number of estimators trained by base_estimator!".format(l_reg))

        if (l_reg > 0 and (regularizer == "none" or regularizer is None)):
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is None. Ignoring l_ensemble_reg!".format(l_reg))
            l_reg = 0
            
        if (l_reg == 0 and (regularizer != "none" and regularizer is not None)):
            print("WARNING: You set l_ensemble_reg to 0, but choose regularizer {}.".format(regularizer))

        if step_size < 0:
            print("WARNING: You supplied a negative step size of {}. Do you want to de-optimize?".format(step_size))

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.max_depth = max_depth
        self.seed = seed
        self.normalize_weights = normalize_weights
        self.burnin_steps = burnin_steps
        self.max_features = max_features
        self.loss = loss
        self.step_size = step_size
        self.optimizer = optimizer
        self.tree_init_mode = tree_init_mode
        self.regularizer = "none" if regularizer is None else regularizer
        self.l_reg = l_reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.out_path = out_path
        self.bootstrap = bootstrap 

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

    def weights(self):
        assert self.model is not None, "Call fit before calling weights!"

        return self.model.weights()

    def num_trees(self):
        assert self.model is not None, "Call fit before calling num_trees!"

        return self.model.num_trees()

    def num_nodes(self):
        assert self.model is not None, "Call fit before calling num_nodes!"

        return self.model.num_nodes()

    def fit(self, X, y, sample_weight = None):

        self.model = COSE(
            len(unique_labels(y)), 
            self.max_depth,
            self.seed,
            self.normalize_weights,
            self.burnin_steps,
            self.max_features,
            self.loss,
            self.step_size,
            self.optimizer,
            self.tree_init_mode, 
            self.regularizer,
            self.l_reg
        )

        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        if self.batch_size > X.shape[0]:
            self.batch_size = X.shape[0]

        self.X_ = X
        self.y_ = y
        
        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(X, y, self.batch_size, shuffle = True, with_replacement = self.bootstrap) 

            acc_sum = 0
            loss_sum = 0
            time_sum = 0
            trees_sum = 0
            nodes_sum = 0
            batch_cnt = 0

            with tqdm(total=X.shape[0], ncols=150, disable = not self.verbose) as pbar:
                for batch in mini_batches: 
                    data, target = batch 
                    
                    output = self.predict_proba(data)

                    # Update Model                    
                    start_time = time.time()
                    self.model.next(data,target)
                    batch_time = time.time() - start_time
                    
                    # Compute the appropriate loss. 
                    if self.loss == "mse":
                        target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
                        loss = (output - target_one_hot) * (output - target_one_hot)
                    elif self.loss == "cross-entropy":
                        target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
                        p = softmax(output, axis=1)
                        loss = -target_one_hot*np.log(p + 1e-7)
                    elif self.loss == "hinge2":
                        target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(self.n_classes_)] for y in target] )
                        zeros = np.zeros_like(target_one_hot)
                        loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
                    else:
                        raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.loss)

                    # Compute the appropriate ensemble_regularizer
                    if self.regularizer == "L0":
                        loss = np.mean(loss) + self.l_reg * np.linalg.norm(self.weights(),0)
                    elif self.regularizer == "L1":
                        loss = np.mean(loss) + self.l_reg * np.linalg.norm(self.weights(),1)
                    else:
                        loss = np.mean(loss) 

                    acc_sum += (target == output.argmax(axis=1)).sum() / data.shape[0]

                    loss_sum += loss
                    time_sum += batch_time
                    trees_sum += self.num_trees()
                    nodes_sum += self.num_nodes()

                    batch_cnt += 1
                    pbar.update(data.shape[0])
                    
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} itime {:2.4f} ntrees {:2.4f} nnodes {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        loss_sum / batch_cnt,
                        acc_sum / batch_cnt,
                        time_sum / batch_cnt,
                        trees_sum / batch_cnt,
                        nodes_sum / batch_cnt
                    )
                    pbar.set_description(desc)
                
                if self.out_path is not None:
                    metrics = {"accuracy":acc_sum / batch_cnt,"loss":loss_sum / batch_cnt, "itime":time_sum / batch_cnt, "ntrees":trees_sum / batch_cnt, "nnodes":nodes_sum / batch_cnt}
                    pickle.dump(metrics, gzip.open(os.path.join(self.out_path, "epoch_{}.pk.gz".format(epoch)), "wb"))
