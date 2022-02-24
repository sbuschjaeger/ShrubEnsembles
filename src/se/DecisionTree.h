#ifndef TREE_H
#define TREE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <queue>
#include <optional>
#include <string_view>
#include <memory>
#include <stdexcept>

#include "Datatypes.h"
#include "Optimizer.h"
#include "Losses.h"
#include "DecisionTree.h"

namespace DT {
    enum TREE_INIT {TRAIN, RANDOM};
}

class Node {
public:
    data_t threshold;
    unsigned int feature;
    unsigned int left, right;
    bool left_is_leaf, right_is_leaf;

    // I want to make sure that these objects are only moved and never copied. I expect the code below to not 
    // use any copy c'tors, but for safe measures we delete the copy constructor entirely.
    Node(const Node&) = default; //delete;
    Node() = default;
    Node(Node &&) = default;

    unsigned int num_bytes() const {
        return sizeof(*this);
    }
};

/**
 * @brief  The main reason why this interface exists, is because it makes class instansiation a little easier for the Pythonbindings. 
 * @note   
 * @retval None
 */
class Tree {
public:

    virtual void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) = 0;

    virtual std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) = 0;

    virtual unsigned int num_bytes() const = 0;

    virtual unsigned int num_nodes() const = 0;

    virtual void set_nodes(std::vector<Node> &nodes) = 0;

    virtual void set_leafs(std::vector<internal_t> &leafs) = 0;

    virtual ~Tree() { }
};

template <DT::TREE_INIT tree_init, OPTIMIZER::OPTIMIZER_TYPE tree_opt>
class DecisionTree : public Tree {

template <LOSS::TYPE friend_loss_type, OPTIMIZER::OPTIMIZER_TYPE friend_opt, OPTIMIZER::OPTIMIZER_TYPE friend_tree_opt, DT::TREE_INIT friend_tree_init>
friend class ShrubEnsemble;

private:
    std::vector<Node> nodes;
    std::vector<internal_t> leafs;
    unsigned int n_classes;
    unsigned int max_depth;
    unsigned int max_features;
    unsigned long seed;

    OPTIMIZER::Optimizer<tree_opt,OPTIMIZER::STEP_SIZE_TYPE::CONSTANT> optimizer;

    inline unsigned int leaf_index(std::vector<data_t> const &x) const {
        unsigned int idx = 0;

        // On small datasets / batchs there might be no node fitted. In this case we only have leaf nodes
        if (nodes.size() > 0) {
            while(true){
                auto const & n = nodes[idx];
                if (x[n.feature] <= n.threshold) {
                    idx = nodes[idx].left;
                    if (n.left_is_leaf) break;
                } else {
                    idx = nodes[idx].right;
                    if (n.right_is_leaf) break;
                }
            }
        } 
        return idx;
    }

     /**
     * @brief  Compute a random split for the given data. This algorithm has O(d * log d + d * N) runtime in the worst case, but should usually run in O(d * log d + N), where N is the number of examples and d is the number of features.
     * This implementation ensures that the returned split splits the data so that each child is non-empty if applied to the given data (at-least one example form X is routed towards left and towards right). The only exception occurs if X is empty or contains one example. In this case we return feature 0 with threshold 1. Threshold-values are placed in the middle between two samples. 
     * @note   
     * @param  &X: The example-set which is used to compute the splitting
     * @param  &Y: The label-set which is used to compute the splitting
     * @param  n_classes: The number of classes
     * @retval The best split as a std::pair<data_t, unsigned int>(best_threshold, best_feature) where the first entry is the threshold and the second entry the feature index.
     */
    static std::optional<std::pair<data_t, unsigned int>> random_split(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, std::vector<unsigned int> const & idx, std::mt19937 &gen) {
        // At-least 2 points are required for splitting.
        // Technically this check is unncessary since we stop tree construction once there is only one label in the data which is always the case 
        // if we have 0 or 1 examples. For safea measure we keep this check alive however.
        if (idx.size() < 2) {
            return std::nullopt;
        }

        // We want to split at a random feature. However, we also want to ensure that the left / right child receive at-least one example with this random
        // split. Sometimes there are features which cannot ensure this (e.g. a binary features where all elements are '1'). Thus, we iterate over a random permutation of features 
        // and return as soon as we find a valid split
        std::vector<unsigned int> features(X[0].size());
        std::iota(std::begin(features), std::end(features), 0); 
        std::shuffle(features.begin(), features.end(), gen);
 
        for (auto const & f: features) {
            // We need to find the next smallest and next biggest value of the data to ensure that left/right will receive at-least 
            // one example. This is a brute force implementation in O(N)
            auto ifirst = idx[0];
            auto isecond = idx[1];

            data_t smallest, second_smallest;
            if(X[ifirst][f] < X[isecond][f]){
                smallest = X[ifirst][f];
                second_smallest = X[isecond][f];
            } else {
                smallest = X[isecond][f];
                second_smallest = X[ifirst][f];
            }

            data_t biggest, second_biggest;
            if(X[ifirst][f] > X[isecond][f]){
                biggest = X[ifirst][f];
                second_biggest = X[isecond][f];
            } else {
                biggest = X[isecond][f];
                second_biggest = X[ifirst][f];
            }

            for (unsigned int j = 2; j < idx.size(); ++j) {
                auto i = idx[j];
                if(X[i][f] > smallest ) { 
                    second_smallest = smallest;
                    smallest = X[i][f];
                } else if(X[i][f] < second_smallest){
                    second_smallest = X[i][f];
                }

                if(X[i][f] > biggest ) { 
                    second_biggest = biggest;
                    biggest = X[i][f];
                } else if(X[i][f] > second_biggest){
                    second_biggest = X[i][f];
                }
            }

            // This is not a valid split if we cannot ensure that the left / right child receive at-least one example.
            if (second_smallest == smallest || second_biggest == biggest) continue;
            std::uniform_real_distribution<> fdis(second_smallest, second_biggest); 

            // So usually I would expect the following line to work, but for some reason it does not. Is this a gcc bug?
            //return std::make_pair<data_t, unsigned int>(static_cast<data_t>(fdis(gen)), f);
            int ftmp = f;
            return std::optional<std::pair<data_t, unsigned int>>{std::make_pair<data_t, unsigned int>(static_cast<data_t>(fdis(gen)), ftmp)};
        }

        return std::nullopt;
    }

    /**
     * @brief  Compute the weighted gini score for the given split. Weighted means here, that we weight the individual gini scores of left and right with the proportion of data in each child node. This leads to slightly more balanced splits.
     * @note   
     * @param  &left: Class-counts for the left child
     * @param  &right: Class-counts for the right child.
     * @retval The weighted gini score.
     */
    static data_t gini(std::vector<unsigned int> const &left, std::vector<unsigned int> const &right) {
        unsigned int sum_left = std::accumulate(left.begin(), left.end(), data_t(0));
        unsigned int sum_right = std::accumulate(right.begin(), right.end(), data_t(0));

        data_t gleft = 0;
        for (auto const l : left) {
            gleft += (static_cast<data_t>(l) / sum_left) * (static_cast<data_t>(l) / sum_left);
        }
        gleft = 1.0 - gleft;

        data_t gright = 0;
        for (auto const r : right) {
            gright += (static_cast<data_t>(r) / sum_right) * (static_cast<data_t>(r) / sum_right);
        }
        gright = 1.0 - gright;

        return sum_left / static_cast<data_t>(sum_left + sum_right) * gleft + sum_right /  static_cast<data_t>(sum_left + sum_right) * gright;
    }
    
    /**
     * @brief  Compute the best split for the given data. This algorithm has O(d * N log N) runtime, where N is the number of examples and d is the number of features.
     * This implementation ensures that the returned split splits the data so that each child is non-empty if applied to the given data (at-least one example form X is routed towards left and towards right). The only exception occurs if X is empty or contains one example. In this case we return feature 0 with threshold 1. Threshold-values are placed in the middle between two samples. 
     * If two splits are equally good, then the first split is chosen. Note that this introduces a slight bias towards the first features. 
     * @note   
     * @param  &X: The example-set which is used to compute the splitting
     * @param  &Y: The label-set which is used to compute the splitting
     * @param  n_classes: The number of classes
     * @retval The best split as a std::pair<data_t, unsigned int>(best_threshold, best_feature) where the first entry is the threshold and the second entry the feature index.
     */
    static std::optional<std::pair<data_t, unsigned int>> best_split(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, std::vector<unsigned int> & idx, long n_classes, unsigned int max_features, std::mt19937 &gen) {
        // At-least 2 points are required for splitting.
        // Technically this check is unncessary since we stop tree construction once there is only one label in the data which is always the case 
        // if we have 0 or 1 examples. For safea measure we keep this check alive however.
        if (idx.size() < 2) {
            return std::nullopt;
        }

        unsigned int n_data = idx.size();
        unsigned int n_features = X[0].size();

        data_t overall_best_gini = 0;
        unsigned int overall_best_feature = 0;
        data_t overall_best_threshold = 0;
        bool split_set = false;

        // Sometimes multiple features have equally good splits (= same gini score). Thus, 
        // we terate over the features in a random order to promote some diversity. Additionally,
        // we also evalute at most max_feature features and return the best unless we were not able
        // to find a suitable feature yet. In that case, we keep looking.
        // This matches the approach SKLearn implements
        std::vector<unsigned int> features(n_features);
        std::iota(features.begin(),features.end(), 0); 
        std::shuffle(features.begin(), features.end(), gen);

        // Prepare class statistics
        std::vector<unsigned int> left_cnts(n_classes);
        std::vector<unsigned int> right_cnts(n_classes);

        unsigned int fcnt = 0;
        for (auto i: features) {
            // In order to compute the best spliting threshold for the current feature we need to evaluate every possible split value.
            // These can be up to n_data - 1 points and for each threshold we need to evaluate if they belong to the left or right child. 
            // The naive implementation thus require O(n_data**2) runtime. We use a slightly more optimized version which requires O(n_data * log n_data). 
            // To do so, we first the examples according to their feature values and compute the initial statistics for the left/right child. Then, we gradually 
            // move the split-threshold to the next value and onyl update the statistics.

            // The data is always accessed indirectly via the idx array sinc this array contains all the indices of the data used 
            // for building the current node. Thus, sort this index wrt. to the current feature.
            std::sort(idx.begin(), idx.end(), [&X, i](unsigned int i1, unsigned int i2){return X[i1][i] < X[i2][i];});

            // Re-set class statistics
            std::fill(left_cnts.begin(), left_cnts.end(), 0);
            std::fill(right_cnts.begin(), right_cnts.end(), 0);
            
            bool first = true;
            unsigned int begin = 0; 
            data_t best_threshold;

            unsigned int jfirst = idx[0];
            for (unsigned int j = 0; j < n_data; ++j) {
                auto jidx = idx[j]; 
            
                if (X[jidx][i] == X[jfirst][i]) {
                    left_cnts[Y[jidx]] += 1;
                } else {
                    if (first) {
                        best_threshold = X[jfirst][i] / 2.0 + X[jidx][i] / 2.0;
                        //best_threshold = 0.5 * (f_values[0].first + f_values[j].first); 
                        first = false;
                        begin = j;
                    }
                    right_cnts[Y[jidx]] += 1;
                }
            }
            
            if (first) {
                // We never choose a threshold which means that X[idx[0]][i] = X[idx[1]][i] = ... = X[idx[end]][i]. 
                // This will not give us a good split, so ignore this feature
                continue;
            }
            // Compute the corresponding gini score 
            data_t best_gini = gini(left_cnts, right_cnts);

            // Repeat what we have done above with the initial scanning, but now update left_cnts / right_cnts appropriately.
            unsigned int j = begin;

            while(j < n_data) {
                auto lidx = idx[j]; 

                do {
                    left_cnts[Y[idx[j]]] += 1;
                    right_cnts[Y[idx[j]]] -= 1;
                    ++j;
                } while(j < n_data && X[idx[j]][i] == X[lidx][i]);
                
                if (j >= n_data) break;
 
                data_t cur_gini = gini(left_cnts, right_cnts);
                data_t threshold = X[lidx][i] / 2.0 + X[idx[j]][i] / 2.0;
                if (cur_gini < best_gini) {
                    best_gini = cur_gini;
                    best_threshold = threshold;
                    //best_threshold = 0.5 * (f_values[j].first + f_values[j + 1].first);
                }
            }

            // Check if we not have already select a split or if this split is better than the other splits we found so far.
            // If so, then set this split
            if (!split_set || best_gini < overall_best_gini) {
                overall_best_gini = best_gini;
                overall_best_feature = i;
                overall_best_threshold = best_threshold;
                split_set = true;
            } 

            // Evaluate at most max_features, but keep looking for splits if we haven found a valid one yet
            fcnt += 1;
            if (fcnt >= max_features && split_set) break;
        }

        if (!split_set) {
            return std::nullopt;
        } else {
            return std::optional<std::pair<data_t, unsigned int>>{std::make_pair(overall_best_threshold, overall_best_feature)};
        }
    }

    void make_leaf(int pid, std::vector<internal_t> &preds, bool is_left) {
        // Normalize the leaf predictions to be proper probabilities (sum to 1)
        data_t sum = std::accumulate(preds.begin(), preds.end(), internal_t(0.0));
        if (sum > 0) {
            std::transform(preds.begin(), preds.end(), preds.begin(), [sum](auto& c){return 1.0/sum*c;});
        } else {
            std::fill_n(preds.begin(), n_classes, internal_t(1.0 / n_classes));
        }

        if (pid >= 0) {
            auto & parent = nodes[pid];
            if (is_left) {
                parent.left_is_leaf = true;
                parent.left = leafs.size();
            } else {
                parent.right_is_leaf = true;
                parent.right = leafs.size();
            }
        }
        leafs.insert(leafs.end(), preds.begin(), preds.end());
    }

public:

    DecisionTree(unsigned int n_classes, unsigned int max_depth, unsigned int max_features, unsigned long seed, internal_t step_size) : n_classes(n_classes),max_depth(max_depth),max_features(max_features),seed(seed),optimizer(step_size) {}

    unsigned int num_bytes() const {
        unsigned int node_size = 0;
        
        for (auto const &n : nodes) {
            node_size += n.num_bytes();
        }

        return sizeof(*this) + node_size + sizeof(internal_t) * leafs.size() + optimizer.num_bytes();
    }

    std::vector<std::vector<internal_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> preds(X.size());
        for (unsigned int i = 0; i < X.size(); ++i) {
            //preds[i] = nodes[node_index(X[i])].preds;
            //data_t const * const node_preds = nodes[node_index(X[i])].preds.get();
            internal_t const * const node_preds = &leafs[leaf_index(X[i])]; //.preds.get();
            preds[i].assign(node_preds, node_preds + n_classes);
        }
        return preds;
    }

    unsigned int num_nodes() const {
        return nodes.size() + int(leafs.size() / n_classes);
    }

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        std::vector<unsigned int> idx(X.size());
        std::iota(std::begin(idx), std::end(idx), 0);
        this->fit(X,Y,idx);
    }

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, std::vector<unsigned int> & idx) {
        /**
         *  For my future self I tried to make the code somewhat understandable while begin reasonably fast / optimized. 
         *  For training the tree we follow the "regular" top-down approach in which we expand each node by two child nodes. The current set of 
         *  nodes-to-be-expanded are kept in a simple queue. The struct below is used to keep track of the current to-be-expanded node, its parent, if its a 
         *  left (or right) child as well as the data on that node. Originally, I used std::vector<std::vector<data_t>> / std::vector<unsigned int> inside
         *  this struct to keep track of the corresponding parts of the data, but this resulted in a large overhead (malloc / free) when splitting large 
         *  datasets. Thus I decided to only store indices to each data point and not copy the whole data. This leads to more indirect memory accesses later 
         *  when computing the best_split but this approach seems to quicker anyhow.  
         * */
        struct TreeExpansion {
            std::vector<unsigned int> idx;
            int parent;
            bool left;
            unsigned int depth;

            // I want to make sure that these objects are only moved and never copied. I expect the code below to not 
            // use any copy c'tors, but for safe measures we delete the copy constructor entirely.
            TreeExpansion(const TreeExpansion&) = delete;
            TreeExpansion() = default;
            TreeExpansion(TreeExpansion &&) = default;
        };
        if (max_features == 0) max_features = X[0].size();

        std::queue<TreeExpansion> to_expand; 
        TreeExpansion root;
        root.idx = std::move(idx);
        root.parent = -1;
        root.left = false;
        root.depth = 0;
        to_expand.push(std::move(root));

        std::mt19937 gen(seed);

        // TODO: Maybe reserve some size in nodes?
        while(to_expand.size() > 0) {
            unsigned int cur_idx = nodes.size();
            auto exp = std::move(to_expand.front());
            to_expand.pop();

            std::vector<internal_t> preds(n_classes, 0.0);
            for (auto i : exp.idx) {
                preds[Y[i]]++;
            }
            // nodes[cur_idx].preds = std::make_unique<internal_t []>(n_classes);

            // Calculate class statistics and check if its a pure node with one class (= leaf node)
            // std::fill_n(nodes[cur_idx].preds.get(), n_classes, 0);
            // for (auto i : exp.idx) {
            //     nodes[cur_idx].preds.get()[Y[i]]++;
            // }

            bool is_leaf = false;
            for (unsigned int i = 0; i < n_classes; ++i) {
                if (preds[i] == exp.idx.size()) {
                    is_leaf = true;
                    break;
                }
            }

            if (is_leaf || (max_depth > 0 && exp.depth >= max_depth)) {
                // Either this node is pure or we reached the max_depth
                // Thus we make this node a leaf and stop building this sub-tree
                this->make_leaf(exp.parent, preds, exp.left);
            } else {
            
                // Compute a suitable split
                std::optional<std::pair<data_t, unsigned int>> split;
                if constexpr (tree_init == DT::TREE_INIT::TRAIN) {
                    split = best_split(X, Y, exp.idx, n_classes, max_features, gen);
                } else {
                    split = random_split(X, Y, exp.idx, gen);
                }

                if (split.has_value()) {
                    // A suitable split as been found
                    // To mitigate costly copy operations of the entire node, we first add it do the vector and then work
                    // on the reference via nodes[cur_idx]
                    nodes.push_back(Node());
                    if (exp.parent >= 0) {
                        if (exp.left) {
                            nodes[exp.parent].left = cur_idx;
                            nodes[exp.parent].left_is_leaf = false;
                        } else {
                            nodes[exp.parent].right = cur_idx;
                            nodes[exp.parent].right_is_leaf = false;
                        }
                    }

                    auto t = split.value().first;
                    auto f = split.value().second;
                    nodes[cur_idx].feature = f;
                    nodes[cur_idx].threshold = t;

                    // We do not need to store the predictions in inner nodes. Thus delete them here
                    // If we want to perform post-pruning at some point we should probably keep these statistics
                    //nodes[cur_idx].preds.reset();

                    TreeExpansion exp_left;
                    exp_left.parent = cur_idx;
                    exp_left.left = true;
                    exp_left.depth = exp.depth + 1;

                    TreeExpansion exp_right;
                    exp_right.parent = cur_idx;
                    exp_right.left = false;
                    exp_right.depth = exp.depth + 1;

                    // Split the data and expand the tree construction
                    for (auto i : exp.idx) {
                        if (X[i][f] <= t) {
                            exp_left.idx.push_back(i);
                        } else {
                            exp_right.idx.push_back(i);
                        }
                    }

                    to_expand.push(std::move(exp_left));
                    to_expand.push(std::move(exp_right));
                } else {
                    // For some reason we were not able to find a suitable split (std::nullopt was returned). 
                    // Thus we make this node a leaf and stop building this sub-tree
                    this->make_leaf(exp.parent, preds, exp.left);
                }
            }
        }
    }

    void set_nodes(std::vector<Node> &new_nodes) {
        nodes = std::move(nodes);
        optimizer.reset();
    }

    void set_leafs(std::vector<internal_t> &new_leafs) {
        leafs = std::move(leafs);
        optimizer.reset();
    }
};

class DecisionTreeClassifier {
private:
	Tree * tree = nullptr;

public:

    DecisionTreeClassifier(
        unsigned int max_depth, 
        unsigned int n_classes, 
        unsigned int max_features,
        unsigned long seed, 
        internal_t step_size,
        const std::string tree_init_mode, 
        const std::string tree_optimizer
    ) { 

        // Yeha this is ugly and there is probably clever way to do this with C++17/20, but this was quicker to code and it gets the job done.
        // Also, lets be real here: There is only a limited chance more init/next modes are added without much refactoring of the whole project
        if (tree_init_mode == "random" && tree_optimizer == "sgd") {
            tree = new DecisionTree<DT::TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::SGD>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "random" && tree_optimizer == "adam") {
            tree = new DecisionTree<DT::TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::ADAM>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "random" && tree_optimizer == "none") {
            tree = new DecisionTree<DT::TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::NONE>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "train" && tree_optimizer == "sgd") {
            tree = new DecisionTree<DT::TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::SGD>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "train" && tree_optimizer == "adam") {
            tree = new DecisionTree<DT::TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::ADAM>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "train" && tree_optimizer == "none") {
            tree = new DecisionTree<DT::TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::NONE>(n_classes,max_depth,max_features,seed,step_size);
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the three  optimizers {none,sgd,adam} are supported for trees, but you provided a combination of " + tree_init_mode + " and " + tree_optimizer);
        }
    }

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (tree != nullptr) {
            tree->fit(X, Y);
        }
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        if (tree != nullptr) {
            return tree->predict_proba(X);
        } else {
            // TODO Add defaults here? 
            return std::vector<std::vector<data_t>>();
        }
    }
    
    ~DecisionTreeClassifier() {
        if (tree != nullptr) {
            delete tree;
        }
    }

    unsigned int num_bytes() const {
        if (tree != nullptr) {
            return tree->num_bytes();
        } else {
            return 0;
        }
    }

    unsigned int num_nodes() const {
        if (tree != nullptr) {
            return tree->num_nodes();
        } else {
            return 0;
        }
    }

    void set_nodes(std::vector<Node> &nodes) {
        if (tree != nullptr) {
            tree->set_nodes(nodes);
        } 
    }

    void set_leafs(std::vector<internal_t> &leafs) {
        if (tree != nullptr) {
            tree->set_leafs(leafs);
        } 
    }
};

#endif