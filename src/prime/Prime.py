import numpy as np
import numbers
import random
from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
import time
import os
from tqdm import tqdm
import gzip
import pickle
import sys

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from .CPrimeBindings import CPrimeBindings

def to_prob_simplex(x):
    if x is None or len(x) == 0:
        return x
    u = np.sort(x)[::-1]

    l = None
    u_sum = 0
    for i in range(0,len(u)):
        u_sum += u[i]
        tmp = 1.0 / (i + 1.0) * (1.0 - u_sum)
        if u[i] + tmp > 0:
            l = tmp
    
    projected_x = [max(xi + l, 0.0) for xi in x]
    return projected_x

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
class Prime(ClassifierMixin, BaseEstimator):
    """ 

    Attributes
    ----------
    loss : str
        The loss function for training. Should be one of `{"mse", "cross-entropy", "hinge2"}`
    step_size : float or str
        The step_size used for stochastic gradient descent. Can be set to "adaptive" for an adaptive step size. 
    normalize_weights : boolean
        True if nonzero weights should be projected onto the probability simplex, that is they should sum to 1. 
    ensemble_regularizer : str
        The ensemble_regularizer. Should be one of `{None, "L0", "L1", "hard-L0"}`
    l_ensemble_reg : float
        The ensemble_regularizer regularization strength. 
    tree_regularizer : str
        The tree_regularizer. Should be one of `{None,"node"}`
    l_tree_reg : float
        The tree_regularizer regularization strength. 
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
    backend: str
        If set to "c++" then the c++ backend will be used. Otherwise the python implementation is used
    estimators_ : list of objects
        The list of estimators which are used to built the ensemble. Each estimator must offer a predict_proba method.
    estimator_weights_ : np.array of floats
        The list of weights corresponding to their respective estimator in self.estimators_. 
    """

    def __init__(self,
                loss = "cross-entropy",
                step_size = 1e-1,
                ensemble_regularizer = None,
                l_ensemble_reg = 0,  
                tree_regularizer = None,
                l_tree_reg = 0,
                normalize_weights = False,
                update_leaves = False,
                batch_size = 256,
                verbose = False,
                out_path = None,
                seed = None,
                epochs = 1,
                backend = "python",
                additional_tree_options = {
                    "splitter" : "best", 
                    "criterion" : "gini",
                    "max_depth": None
                },
                warmstart = True
        ):

        assert loss in ["mse","cross-entropy","hinge2"], "Currently only {{mse, cross-entropy, hinge2}} loss is supported"
        assert ensemble_regularizer is None or ensemble_regularizer in ["none","L0", "L1", "hard-L0"], "Currently only {{none,L0, L1, hard-L0}} the ensemble regularizer is supported"
        assert l_tree_reg >= 0, "l_tree_reg must be greate or equal to 0"
        assert tree_regularizer is None or tree_regularizer in ["node"], "Currently only {{none, node}} regularizer is supported for tree the regularizer."

        if backend == "c++":
            assert "max_depth" in additional_tree_options and additional_tree_options["max_depth"] > 0, "The C++ backend required a maximum tree depth to be set, but none was given"

        if batch_size is None or batch_size < 1:
            print("WARNING: batch_size should be 2 for PyBiasedProxEnsemble for optimal performance, but was {}. Fixing it for you.".format(batch_size))
            batch_size = 2

        if ensemble_regularizer == "hard-L0" and l_ensemble_reg < 1:
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is hard-L0. In this mode, l_ensemble_reg should be an integer 1 <= l_ensemble_reg <= max_trees where max_trees is the number of estimators trained by base_estimator!".format(l_ensemble_reg))

        if (l_ensemble_reg > 0 and (ensemble_regularizer == "none" or ensemble_regularizer is None)):
            print("WARNING: You set l_ensemble_reg to {}, but regularizer is None. Ignoring l_ensemble_reg!".format(l_ensemble_reg))
            l_ensemble_reg = 0
            
        if (l_ensemble_reg == 0 and (ensemble_regularizer != "none" and ensemble_regularizer is not None)):
            print("WARNING: You set l_ensemble_reg to 0, but choose regularizer {}.".format(ensemble_regularizer))

        if "random_seed" in additional_tree_options:
            print("WARNING: You passed `random_seed` to additional_tree_options. However, the random_seed is defined internally for individual trees. You can control the random seed by setting the `seed` parameter of Prime. I am going to ignore the random_seed parameter passed to additional_tree_options")
            del additional_tree_options["random_seed"]

        if isinstance(step_size, str) and step_size != "adaptive":
            step_size = float(step_size)

        if "tree_init_mode" in additional_tree_options:
            assert additional_tree_options["tree_init_mode"] in ["train", "fully-random", "random"], "Currently only {{train, fully-random, random}} as tree_init_mode is supported"
            self.tree_init_mode = additional_tree_options["tree_init_mode"]
        else:
            self.tree_init_mode = "train"

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.dt_seed = self.seed
        self.step_size = step_size
        self.loss = loss
        self.normalize_weights = normalize_weights
        self.ensemble_regularizer = ensemble_regularizer
        self.l_ensemble_reg = l_ensemble_reg
        self.tree_regularizer = tree_regularizer
        self.l_tree_reg = l_tree_reg
        self.normalize_weights = normalize_weights
        self.estimators_ = [] # Only used if backend is python
        self.estimator_weights_ = [] # Only used if backend is python
        self.update_leaves = update_leaves
        self.additional_tree_options = additional_tree_options
        self.backend = backend

        self.batch_size = batch_size
        self.verbose = verbose
        self.out_path = out_path
        self.epochs = epochs
        self.warmstart = warmstart

    def _individual_proba(self, X):
        ''' Predict class probabilities for each individual learner in the ensemble without considering the weights.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,C)
            The predicted class probabilities for each learner.
        '''
        assert self.estimators_ is not None, "Call fit before calling predict_proba!"
        all_proba = []

        for e in self.estimators_:
            if len(X.shape) < 2:
                proba = np.zeros(shape=(1, self.n_classes_), dtype=np.float32)
                X = X[np.newaxis,:]
            else:
                proba = np.zeros(shape=(X.shape[0], self.n_classes_), dtype=np.float32)

            proba[:, e.classes_.astype(int)] += e.predict_proba(X)
            all_proba.append(proba)

        if len(all_proba) == 0:
            if len(X.shape) < 0:
                return np.zeros(shape=(1, 1, self.n_classes_), dtype=np.float32)
            else:
                return np.zeros(shape=(1, X.shape[0], self.n_classes_), dtype=np.float32)
        else:
            return np.array(all_proba)

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

        if self.backend == "c++":
            if len(X.shape) < 2:
                # The c++ bindings only supports batched data and thus we add the implicit batch dimension via data[np.newaxis,:]
                return np.array(self.model.predict_proba(X[np.newaxis,:]))[0]
            else:
                return np.array(self.model.predict_proba(X))
        else:
            # Check is fit had been called
            # check_is_fitted(self, ['X_', 'y_'])

            # Input validation
            # X = check_array(X)
            if (len(self.estimators_)) == 0:
                if len(X.shape) < 0:
                    return 1.0 / self.n_classes_ * np.ones((1, self.n_classes_))
                else:
                    return 1.0 / self.n_classes_ * np.ones((X.shape[0], self.n_classes_))
            else:
                all_proba = self._individual_proba(X)
                scaled_prob = np.array([w * p for w,p in zip(all_proba, self.estimator_weights_)])
                combined_proba = np.sum(scaled_prob, axis=0)
                return combined_proba

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

    def next(self, data, target):
        if self.backend == "c++":
            self.model.next(data, target)
        else:
            if (len(set(target)) > 1):
                # Fit a new tree on the current batch. 

                tree = DecisionTreeClassifier(random_state=self.dt_seed, **self.additional_tree_options)

                self.dt_seed += 1
                tree.fit(data, target)

                # SKlearn stores the raw counts instead of probabilities. For SGD its better to have the 
                # probabilities for numerical stability. 
                # tree.tree_.value is not writeable, but we can modify the values inplace. Thus we 
                # use [:] to copy the array into the normalized array. Also tree.tree_.value has a strange shape
                # (batch_size, 1, n_classes)
                
                tree.tree_.value[:] = tree.tree_.value / tree.tree_.value.sum(axis=(1,2))[:,np.newaxis,np.newaxis]

                self.estimator_weights_.append(0.0)
                self.estimators_.append(tree)
            # else:
            #     # TODO WHAT TO DO IF ONLY ONE LABEL IS IN THE CURRENT BATCH?
            #     pass

            if len(self.estimators_) > 0:
                all_proba = self._individual_proba(data)
                output = np.array([w * p for w,p in zip(all_proba, self.estimator_weights_)]).sum(axis=0)

                # Compute the appropriate loss. 
                if self.loss == "mse":
                    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
                    # loss = (output - target_one_hot) * (output - target_one_hot)
                    loss_deriv = 2 * (output - target_one_hot)
                elif self.loss == "cross-entropy":
                    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(self.n_classes_)] for y in target] )
                    p = softmax(output, axis=1)
                    # loss = -target_one_hot*np.log(p + 1e-7)
                    m = target.shape[0]
                    loss_deriv = softmax(output, axis=1)
                    loss_deriv[range(m),target_one_hot.argmax(axis=1)] -= 1
                elif self.loss == "hinge2":
                    target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(self.n_classes_)] for y in target] )
                    zeros = np.zeros_like(target_one_hot)
                    # loss = np.maximum(1.0 - target_one_hot * output, zeros)**2
                    loss_deriv = - 2 * target_one_hot * np.maximum(1.0 - target_one_hot * output, zeros) 
                else:
                    raise "Currently only the losses {{cross-entropy, mse, hinge2}} are supported, but you provided: {}".format(self.loss)

            
                # Compute the gradient for the loss
                directions = np.mean(all_proba*loss_deriv,axis=(1,2))
                
                # Compute the gradient for the tree regularizer
                if self.tree_regularizer:
                    node_deriv = self.l_tree_reg * np.array([ est.tree_.node_count for est in self.estimators_])
                else:
                    node_deriv = 0

                # Perform the gradient step. Note that L0 / L1 regularizer is performed via the prox operator 
                # and thus performed _after_ this update.
                if self.step_size == "adaptive":
                    step_size = 1.0 / (len(self.estimators_) + 1.0)
                else:
                    step_size = self.step_size

                tmp_w = self.estimator_weights_ - step_size*directions - step_size*node_deriv
                
                if self.update_leaves:
                    for i, h in enumerate(self.estimators_):
                        tree_grad = (self.estimator_weights_[i] * loss_deriv)[:,np.newaxis,:]
                        # find idx
                        idx = h.apply(data)
                        h.tree_.value[idx] = h.tree_.value[idx] - step_size * tree_grad[:,:,h.classes_.astype(int)]

                # Compute the prox step. 
                if self.ensemble_regularizer == "L0":
                    tmp = np.sqrt(2 * self.l_ensemble_reg * step_size)
                    tmp_w = np.array([0 if abs(w) < tmp else w for w in tmp_w])
                elif self.ensemble_regularizer == "L1":
                    sign = np.sign(tmp_w)
                    tmp_w = np.abs(tmp_w) - step_size*self.l_ensemble_reg
                    tmp_w = sign*np.maximum(tmp_w,0)
                elif self.ensemble_regularizer == "hard-L0":
                    top_K = np.argsort(tmp_w)[-self.l_ensemble_reg:]
                    tmp_w = np.array([w if i in top_K else 0 for i,w in enumerate(tmp_w)])

                # If set, normalize the weights. Note that we use the support of tmp_w for the projection onto the probability simplex
                # as described in http://proceedings.mlr.press/v28/kyrillidis13.pdf
                # Thus, we first need to extract the nonzero weights, project these and then copy them back into corresponding array
                # print("PRE PROX: ", tmp_w)
                if self.normalize_weights and len(tmp_w) > 0:
                    nonzero_idx = np.nonzero(tmp_w)[0]
                    nonzero_w = tmp_w[nonzero_idx]
                    nonzero_w = to_prob_simplex(nonzero_w)
                    self.estimator_weights_ = np.zeros((len(tmp_w)))
                    for i,w in zip(nonzero_idx, nonzero_w):
                        self.estimator_weights_[i] = w
                else:
                    self.estimator_weights_ = tmp_w
                
                new_est = []
                new_w = []
                for h, w in zip(self.estimators_, self.estimator_weights_):
                    if w > 0:
                        new_est.append(h)
                        new_w.append(w)

                self.estimators_ = new_est
                self.estimator_weights_ = new_w


    def num_bytes(self):
        self_size = sys.getsizeof(self.step_size) + sys.getsizeof(self.loss) + sys.getsizeof(self.normalize_weights) + sys.getsizeof(self.ensemble_regularizer) + sys.getsizeof(self.l_ensemble_reg) + sys.getsizeof(self.tree_regularizer) + sys.getsizeof(self.l_tree_reg) + sys.getsizeof(self.normalize_weights) + sys.getsizeof(self.dt_seed) + sys.getsizeof(self.update_leaves) + sys.getsizeof(self.additional_tree_options) + sys.getsizeof(self.backend) + sys.getsizeof(self.batch_size) + sys.getsizeof(self.verbose) + sys.getsizeof(self.out_path) + sys.getsizeof(self.epochs) + sys.getsizeof(self.warmstart) + sys.getsizeof(self.tree_init_mode)

        if self.backend == "c++":
            # model = self.model
            # self.model = None
            # p = pickle.dumps(self)
            # size = sys.getsizeof(p)
            # self.model = model
            return self_size + self.model.num_bytes()
        else:
            sk_size = 0
            for e in self.estimators_:
                p = pickle.dumps(e)
                sk_size += sys.getsizeof(p)
            return self_size + sys.getsizeof(self.estimator_weights_) + sk_size

    def num_trees(self):
        if self.backend == "c++":
            return self.model.num_trees()
        else:
            return np.count_nonzero(self.estimator_weights_)

    def num_nodes(self):
        if self.backend == "c++":
            return self.model.num_nodes()
        else:
            return sum( [ est.tree_.node_count if w != 0 else 0 for w, est in zip(self.estimator_weights_, self.estimators_)] )

    def fit(self, X, y, sample_weight = None):
        if self.backend == "c++":
            if self.step_size == "adaptive":
                step_size_mode = "adaptive"
                step_size = 0
            else:
                step_size_mode = "constant"
                step_size = float(self.step_size)

            if self.update_leaves:
                tree_update_mode = "gradient"
            else:
                tree_update_mode = "none"
            
            ensemble_regularizer = "none" if self.ensemble_regularizer is None else str(self.ensemble_regularizer)
            tree_regularizer = "none" if self.tree_regularizer is None else str(self.tree_regularizer)

            self.model = CPrimeBindings(
                len(unique_labels(y)), 
                self.additional_tree_options["max_depth"],
                self.seed,
                self.normalize_weights,
                self.loss,
                step_size,
                step_size_mode,
                ensemble_regularizer,
                float(self.l_ensemble_reg),
                tree_regularizer,
                float(self.l_tree_reg),
                self.tree_init_mode, 
                tree_update_mode
            )

        if self.warmstart:
            if self.ensemble_regularizer == "hard-L0":
                n_estimators = self.l_ensemble_reg
            else:
                # TODO Add a parameter for this?
                n_estimators = 32

            # TODO This can be done in parallel
            for _ in range(n_estimators):
                idx = np.random.choice(range(0,len(X)), replace = True, size = self.batch_size)
                data, target = X[idx],y[idx]
                w = 1.0 / n_estimators

                if self.backend == "c++":
                    self.model.add_tree(data, target, w)
                else:
                    tree = DecisionTreeClassifier(random_state=self.dt_seed, **self.additional_tree_options)

                    self.dt_seed += 1
                    tree.fit(data, target)
                
                    tree.tree_.value[:] = tree.tree_.value / tree.tree_.value.sum(axis=(1,2))[:,np.newaxis,np.newaxis]

                    self.estimators_.append(tree)
                    self.estimator_weights_.append(w)
        
        # TODO respect sample weights in loss
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = self.n_classes_
        
        self.X_ = X
        self.y_ = y
        
        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(X, y, self.batch_size, True, False) 

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
                    self.next(data,target)
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
                    if self.ensemble_regularizer == "L0":
                        loss = np.mean(loss) + self.l_ensemble_reg * np.linalg.norm(self.estimator_weights_,0)
                    elif self.ensemble_regularizer == "L1":
                        loss = np.mean(loss) + self.l_ensemble_reg * np.linalg.norm(self.estimator_weights_,1)
                    else:
                        loss = np.mean(loss) 
                    
                    # Compute the appropriate tree_regularizer
                    if self.tree_regularizer == "node":
                        loss += self.l_tree_reg * np.sum( [ (w * est.tree_.node_count) for w, est in zip(self.estimator_weights_, self.estimators_)] )

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
