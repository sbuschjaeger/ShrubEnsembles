import numpy as np
import random
from scipy.special import softmax
from sklearn.utils.multiclass import unique_labels
import os
from tqdm import tqdm
import gzip
import pickle

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin

from se.CShrubEnsembles import CMASE

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
class MASE(ClassifierMixin, BaseEstimator):

    def __init__(self,
        max_depth = 5,
        seed = 12345,
        burnin_steps = 0,
        max_features = 0,
        loss = "mse",
        step_size = 1e-2,
        optimizer = "sgd",
        tree_init_mode = "train",
        n_trees = 32, 
        n_rounds = 5,
        n_worker = 5,
        batch_size_per_worker = 0,
        bootstrap = True,
        verbose = False,
        out_path = None,
        sample_engine = "python",
        batch_size = None
    ):

        # TODO set n_jobs for openmp?
        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"

        assert max_depth >= 0, "max_depth must be >= 0. Use max_depth = 0 if you want to train unlimited trees. You supplied: {}".format(max_depth)

        assert max_features >= 0, "max_features must be >= 0. Use max_features = 0 if you want to evaluated all splits. You supplied: {}".format(max_features)

        assert sample_engine in ["python", "c++"], "Currently only {{python, c++}} can be used as the sample_egine"

        if sample_engine == "c++" and batch_size is not None:
            print("WARNING: You supplied sample_engine = `c++', but set batch_size != None. The c++ sample engine does not support batch_sizes and will always use the entire dataset to perform SGD!")

        if sample_engine == "python":
            assert batch_size is None or batch_size > 0, "You supplied sample_engine=`python`. In this case choose batch_size > 0 or batch_size = None (which uses the entire dataset)."

        # assert n_rounds > 0, "Epochs should be at least 1, but you gave {}".format(n_rounds)

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
        self.burnin_steps = burnin_steps
        self.max_features = max_features
        self.loss = loss
        self.step_size = step_size
        self.optimizer = optimizer
        self.tree_init_mode = tree_init_mode
        self.n_trees = n_trees
        self.batch_size_per_worker = batch_size_per_worker
        self.n_rounds = n_rounds
        self.n_worker = n_worker
        self.verbose = verbose
        self.out_path = out_path
        self.bootstrap = bootstrap 
        self.sample_engine = sample_engine
        self.batch_size = batch_size

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

        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        if self.batch_size_per_worker > X.shape[0]:
            self.batch_size_per_worker = X.shape[0]

        self.X_ = X
        self.y_ = y
        self.model = CMASE(
            len(unique_labels(y)), 
            self.max_depth,
            self.seed,
            self.burnin_steps,
            self.max_features,
            self.loss,
            self.step_size,
            self.optimizer,
            self.tree_init_mode, 
            self.n_trees,
            self.n_worker,
            self.n_rounds,
            self.batch_size_per_worker,
            self.bootstrap
        )

        if self.sample_engine == "c++":
            self.model.fit(X,y)
        else: 
            if self.batch_size is None:
                self.model.init(X,y)
            else:
                Nsample = self.batch_size
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                Xs, Ys = X[indices[0:Nsample]], y[indices[0:Nsample]]
                self.model.init(Xs,Ys)

            with tqdm(total=self.n_rounds, ncols=150, disable = not self.verbose) as pbar:

                for r in range(self.n_rounds):

                    if self.batch_size is None:
                        self.model.next(X,y)
                        Xs, Ys = X,y
                    else:
                        Nsample = self.batch_size
                        indices = np.arange(X.shape[0])
                        np.random.shuffle(indices)
                        Xs, Ys = X[indices[0:Nsample]], y[indices[0:Nsample]]
                        self.model.next(Xs,Ys)
                    
                    if self.verbose or self.out_path is not None:
                        output = self.predict_proba(Xs)

                        # Compute the appropriate loss. 
                        if self.loss == "mse":
                            target_one_hot = np.array( [ [1.0 if yi == i else 0.0 for i in range(self.n_classes_)] for yi in Ys] )
                            loss = (output - target_one_hot) * (output - target_one_hot)
                        elif self.loss == "cross-entropy":
                            target_one_hot = np.array( [ [1.0 if yi == i else 0.0 for i in range(self.n_classes_)] for yi in Ys] )
                            p = softmax(output, axis=1)
                            loss = -target_one_hot*np.log(p + 1e-7)
                        elif self.loss == "hinge2":
                            target_one_hot = np.array( [ [1.0 if yi == i else -1.0 for i in range(self.n_classes_)] for yi in Ys] )
                            zeros = np.zeros_like(target_one_hot)
                            loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
                        else:
                            raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.loss)

                        loss = np.mean(loss) 
                        accuracy = (Ys == output.argmax(axis=1)).sum() / Xs.shape[0]
                        pbar.update(1)
                        
                        desc = '[{}/{}] loss {:2.4f} acc {:2.4f}'.format(
                            r, 
                            self.n_rounds-1, 
                            loss,
                            accuracy
                        )
                        pbar.set_description(desc)
                    
                    if self.out_path is not None:
                        metrics = {"accuracy":accuracy,"loss":loss}
                        pickle.dump(metrics, gzip.open(os.path.join(self.out_path, "round_{}.pk.gz".format(r)), "wb"))
