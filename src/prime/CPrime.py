import numpy as np
import numbers
import random
from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
import time
import os
from tqdm import tqdm

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin

from .CPrimeBindings import CPrimeBindings

# Modified from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
def create_mini_batches(inputs, targets, batch_size, shuffle=False, sliding_window=False):
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        if sliding_window:
            start_idx += 1
        else:
            start_idx += batch_size

        yield inputs[excerpt], targets[excerpt]

# TODO Add Regressor
class CPrime(ClassifierMixin, BaseEstimator):
    """ 

    Attributes
    ----------
    max_depth : int
        Maximum depth of DTs trained on each batch
    step_size : float
        The step_size used for stochastic gradient descent for opt 
    loss : str
        The loss function for training. Should be one of `{"mse", "cross-entropy", "hinge2"}`
    normalize_weights : boolean
        True if nonzero weights should be projected onto the probability simplex, that is they should sum to 1. 
    ensemble_regularizer : str
        The ensemble_regularizer. Should be one of `{None, "L0", "L1", "hard-L1"}`
    l_ensemble_reg : float
        The ensemble_regularizer regularization strength. 
    tree_regularizer : str
        The tree_regularizer. Should be one of `{None,"node"}`
    l_tree_reg : float
        The tree_regularizer regularization strength. 
    init_weight : str, number
        The weight initialization for each new tree. If this is `"max`" then the largest weight across the entire ensemble is used. If this is `"average"` then the average weight  across the entire ensemble is used. If this is a number, then the supplied value is used. 
    batch_size: int
        The batch sized used for SGD
    update_leaves : boolean
        If true, then leave nodes of each tree are also updated via SGD.
    epochs : int
        The number of epochs SGD is run.
    verbose : boolean
        If true, shows a progress bar via tqdm and some statistics
    out_path: str
        If set, stores a file called epoch_$i.npy with the statistics for epoch $i under the given path.
    seed : None or number
        Random seed for tree construction. If None, then the seed 1234 is used.
    estimators_ : list of objects
        The list of estimators which are used to built the ensemble. Each estimator must offer a predict_proba method.
    estimator_weights_ : np.array of floats
        The list of weights corresponding to their respective estimator in self.estimators_. 
    """

    def __init__(self,
                max_depth,
                loss = "cross-entropy",
                step_size = 1e-1,
                ensemble_regularizer = None,
                l_ensemble_reg = 0,  
                tree_regularizer = None,
                l_tree_reg = 0,
                normalize_weights = False,
                init_weight = 0,
                update_leaves = False,
                batch_size = 256,
                verbose = False,
                out_path = None,
                seed = None,
                epochs = None
        ):

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert ensemble_regularizer is None or ensemble_regularizer in ["none","L0", "L1", "hard-L1"], "Currently only {{none,L0, L1, hard-L1}} the ensemble regularizer is supported"
        assert init_weight in ["average","max"] or isinstance(init_weight, numbers.Number), "init_weight should be {{average, max}} or a number"
        assert not isinstance(init_weight, numbers.Number) or (isinstance(init_weight, numbers.Number) and init_weight > 0), "init_weight should be > 0, otherwise it will we removed immediately after its construction."
        assert l_tree_reg >= 0, "l_tree_reg must be greate or equal to 0"
        assert tree_regularizer is None or tree_regularizer in ["node"], "Currently only {{none, node}} regularizer is supported for tree the regularizer."
        
        if batch_size is None or batch_size < 1:
            print("WARNING: batch_size should be 2 for PyBiasedProxEnsemble for optimal performance, but was {}. Fixing it for you.".format(batch_size))
            batch_size = 2

        if ensemble_regularizer == "hard-L1" and l_ensemble_reg < 1:
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is hard-L1. In this mode, l_ensemble_reg should be an integer 1 <= l_ensemble_reg <= max_trees where max_trees is the number of estimators trained by base_estimator!".format(l_ensemble_reg))

        if (l_ensemble_reg > 0 and (ensemble_regularizer == "none" or ensemble_regularizer is None)):
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is None. Ignoring l_ensemble_reg!".format(l_ensemble_reg))
            l_ensemble_reg = 0
            
        if (l_ensemble_reg == 0 and (ensemble_regularizer != "none" and ensemble_regularizer is not None)):
            print("WARNING: You set l_ensemble_reg to 0, but choose regularizer {}.".format(ensemble_regularizer))

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.step_size = step_size
        self.loss = loss
        self.normalize_weights = normalize_weights
        self.init_weight = init_weight
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.tree_regularizer = tree_regularizer
        self.l_tree_reg = l_tree_reg
        self.normalize_weights = normalize_weights
        self.max_depth = max_depth
        self.estimators_ = []
        self.estimator_weights_ = []
        self.dt_seed = self.seed
        self.update_leaves = update_leaves

        self.batch_size = batch_size
        self.verbose = verbose
        self.out_path = out_path # TODO Currently unused, can be removed 
        self.epochs = epochs
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
        assert self.model is not None, "Please call fit before calling predict or predict_proba"

        return self.model.predict_proba(X)

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

    def fit(self, X, y, sample_weight = None):
        # TODO respect sample weights in loss
        X, y = check_X_y(X, y)
        
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        self.X_ = X
        self.y_ = y
        
        if self.init_weight in ["average", "max"]:
            weight_init_mode = self.init_weight
            init_weight = 0
        else:
            weight_init_mode = "constant"
            init_weight = self.init_weight
        
        # TODO Add interface for this 
        tree_init_mode = "train"

        if self.update_leaves:
            tree_update_mode = "gradient"
        else:
            tree_update_mode = "none"
        
        # TODO Add interface for this 
        is_nominal = [False for _ in range(X.shape[1])]

        self.model = CPrimeBindings(
            self.n_classes_, 
            self.max_depth,
            self.seed,
            self.normalize_weights,
            self.loss,
            self.step_size,
            weight_init_mode,
            init_weight,
            is_nominal,
            self.ensemble_regularizer,
            self.l_ensemble_reg,
            self.tree_regularizer,
            self.l_tree_reg,
            tree_init_mode, 
            tree_update_mode
        )

        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(X, y, self.batch_size, False, False) 

            total_time = 0
            total_loss = 0

            # first_batch = True
            example_cnt = 0

            with tqdm(total=X.shape[0], ncols=150, disable = not self.verbose) as pbar:
                for batch in mini_batches: 
                    data, target = batch 
                    
                    # Update Model                    
                    start_time = time.time()
                    batch_loss = self.next(data, target)
                    batch_time = time.time() - start_time

                    total_loss += batch_loss
                    total_time += batch_time
                    example_cnt += data.shape[0]

                    pbar.update(data.shape[0])
                    desc = '[{}/{}] loss {} time {}'.format(
                        epoch, 
                        self.epochs-1, 
                        total_loss / example_cnt, 
                        total_time / example_cnt,
                    )
                    pbar.set_description(desc)
                