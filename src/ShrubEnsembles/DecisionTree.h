#pragma once

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <queue>
#include <optional>
#include <string_view>
#include <memory>
#include <stdexcept>

#include "Tree.h"
#include "Datatypes.h"
#include "Matrix.h"
#include "Losses.h"
// #include "DecisionTree.h"

namespace DT {
    enum INIT {GINI, RANDOM, CUSTOM};
}

template <typename data_t, DT::INIT tree_init>
class DecisionTree : public Tree<data_t> {

static_assert(std::is_arithmetic<data_t>::value, "The data type data_t must be an arithmetic data type. See std::is_arithmetic for more details.");

protected:
    std::vector<Node<data_t>> _nodes;
    std::vector<internal_t> _leaves;
    unsigned int n_classes;
    unsigned int max_depth;
    unsigned int max_features;
    unsigned long seed;

    /**
     * This is a trade-off between compile-time performance and usability. Basically, I wanted to have as much speed as possible (without too crazy optimizations) while also being able to exchange score/distance functions quickly. Hence, I decided to mix compile-time constants via templates as well as some dynamic data structures. The idea is that whenever a user uses an INIT and/or DISTANCE function that is implemented in C++ and available during compile-time, we try to hard-code its usage during compile-time. The most straightforward way to do this would be to simply use a class as a template parameter (e.g., write DistanceDecisionTree<double, MyGiniScoreClass, MyGZIPClass>). This is nice from a C++ perspective, but it does not allow us to implement score/distance functions in Python. Hence, we use a special flag DDT::INIT::CUSTOM / DDT::DISTANCE::CUSTOM to use a "dynamic" version of this class. Inside the code, we use constexpr to check during compile time if we are using e.g., the gini score or a custom score function. If we are using a custom score (or distance), then the _score /_distance fields are accessed. If, however, we are using e.g., the gini score, then we will never access _score and directly call the appropriate scoring function. In this case, the _score member basically lies dormant in our code. The reason for this approach is that accessing std::function has a small overhead. While this usually does not impact the performance much, it can be noticeable when a function gets called very often. Unfortunately, this is the case for the Gini score (and to some degree for the distance function). In a small series of benchmarks, I found that training DTs on datasets with many features and (comparably) fewer examples, there can be a performance penalty of up to 50%. For CIFAR10 datasets (50K examples, 32*32*3 =3072 features) the performance difference was around 10 % on average. 
     * The downside of this approach is, that we have ~ 32 Bytes (tested via godbolt) per std::optional<std::function<..>> construct as overhead in every object. Also, the best_split function now depends on the template parameter and hence cannot be static anymore, increasing the code size. 
     * There might be more room for optimization / making the whole construct nicer, but for regular DTs I also tested a few other methods, including only using std::function and using raw function pointers, which never matched the performance of this approach. In fact, raw function pointers seem to be on par with std::functions here. 
     */
    std::optional<std::function<internal_t(std::vector<unsigned int> const &, std::vector<unsigned int> const &)>> score;

     /**
     * @brief  Compute a random split for the given data. This algorithm has O(d * log d + d * N) runtime in the worst case, but should usually run in O(d * log d + N), where N is the number of examples and d is the number of features.
     * This implementation ensures that the returned split splits the data so that each child is non-empty if applied to the given data (at-least one example form X is routed towards left and towards right). The only exception occurs if X is empty or contains one example. In this case we return feature 0 with threshold 1. Threshold-values are placed in the middle between two samples. 
     * @note   
     * @param  &X: The example-set which is used to compute the splitting
     * @param  &Y: The label-set which is used to compute the splitting
     * @param  n_classes: The number of classes
     * @retval The best split as a std::pair<data_t, unsigned int>(best_threshold, best_feature) where the first entry is the threshold and the second entry the feature index.
     */
    static std::optional<std::pair<data_t, unsigned int>> random_split(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, std::vector<unsigned int> const & idx, std::mt19937 &gen, std::vector<bool> & feature_is_const) {
        // At-least 2 points are required for splitting.
        // Technically this check is unncessary since we stop tree construction once there is only one label in the data which is always the case 
        // if we have 0 or 1 examples. For safea measure we keep this check alive however.
        if (idx.size() < 2) {
            return std::nullopt;
        }

        // We want to split at a random feature. However, we also want to ensure that the left / right child receive at-least one example with this random
        // split. Sometimes there are features which cannot ensure this (e.g. a binary features where all elements are '1'). Thus, we iterate over a random permutation of features 
        // and return as soon as we find a valid split
        std::vector<unsigned int> features(X.cols);
        std::iota(std::begin(features), std::end(features), 0); 
        std::shuffle(features.begin(), features.end(), gen);
 
        for (auto const & f: features) {
            if (feature_is_const[f]) {
                continue;
            }
            // We need to find the next smallest and next biggest value of the data to ensure that left/right will receive at-least 
            // one example. This is a brute force implementation in O(N)
            auto ifirst = idx[0];
            auto isecond = idx[1];

            data_t smallest, second_smallest;
            if(X(ifirst, f) < X(isecond, f)){
                smallest = X(ifirst, f);
                second_smallest = X(isecond, f);
            } else {
                smallest = X(isecond, f);
                second_smallest = X(ifirst, f);
            }

            data_t biggest, second_biggest;
            if(X(ifirst, f) > X(isecond, f)){
                biggest = X(ifirst, f);
                second_biggest = X(isecond, f);
            } else {
                biggest = X(isecond, f);
                second_biggest = X(ifirst, f);
            }

            for (unsigned int j = 2; j < idx.size(); ++j) {
                auto i = idx[j];
                if(X(i,f) > smallest ) { 
                    second_smallest = smallest;
                    smallest = X(i,f);
                } else if(X(i,f) < second_smallest){
                    second_smallest = X(i,f);
                }

                if(X(i,f) > biggest ) { 
                    second_biggest = biggest;
                    biggest = X(i,f);
                } else if(X(i,f) > second_biggest){
                    second_biggest = X(i,f);
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
    static internal_t gini(std::vector<unsigned int> const &left, std::vector<unsigned int> const &right) {
        unsigned int sum_left = std::accumulate(left.begin(), left.end(), 0);
        unsigned int sum_right = std::accumulate(right.begin(), right.end(), 0);

        internal_t gleft = 0;
        for (auto const l : left) {
            gleft += (static_cast<internal_t>(l) / sum_left) * (static_cast<internal_t>(l) / sum_left);
        }
        gleft = 1.0 - gleft;

        internal_t gright = 0;
        for (auto const r : right) {
            gright += (static_cast<internal_t>(r) / sum_right) * (static_cast<internal_t>(r) / sum_right);
        }
        gright = 1.0 - gright;

        return sum_left / static_cast<internal_t>(sum_left + sum_right) * gleft + sum_right /  static_cast<internal_t>(sum_left + sum_right) * gright;
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
    std::optional<std::pair<internal_t, unsigned int>> best_split(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, std::vector<unsigned int> const &idx, long n_classes, unsigned int max_features, std::mt19937 &gen, std::vector<bool> & feature_is_const) {
        // At-least 2 points are required for splitting.
        // Technically this check is unncessary since we stop tree construction once there is only one label in the data which is always the case 
        // if we have 0 or 1 examples. For safea measure we keep this check alive however.
        if (idx.size() < 2) {
            return std::nullopt;
        }

        unsigned int n_data = idx.size();
        unsigned int n_features = X.cols;

        internal_t overall_best_score = 0;
        unsigned int overall_best_feature = 0;
        data_t overall_best_threshold = 0;
        bool split_set = false;

        // Sometimes multiple features have equally good splits (= same gini score). Thus, 
        // we tierate over the features in a random order to promote some diversity. Additionally,
        // we also evalute at most max_feature features and return the best unless we were not able
        // to find a suitable feature yet. In that case, we keep looking.
        // This matches the approach SKLearn implements
        std::vector<unsigned int> features(n_features);
        std::iota(features.begin(),features.end(), 0); 
        std::shuffle(features.begin(), features.end(), gen);

        // Prepare class statistics
        std::vector<unsigned int> left_cnts(n_classes);
        std::vector<unsigned int> right_cnts(n_classes);

        struct Sample {
            data_t xf;
            unsigned int label;
        };

        std::vector<Sample> XTmp(n_data);

        unsigned int fcnt = 0;
        for (auto i: features) {
            if (feature_is_const[i]) {
                continue;
            }
            // In order to compute the best spliting threshold for the current feature we need to evaluate every possible split value.
            // These can be up to n_data - 1 points and for each threshold we need to evaluate if they belong to the left or right child. 
            // The naive implementation thus require O(n_data^2) runtime. We use a slightly more optimized version which requires O(n_data * log n_data). 
            // To do so, we first the examples according to their feature values and compute the initial statistics for the left/right child. Then, we gradually 
            // move the split-threshold to the next value and onyl update the statistics.

            // The data is always accessed indirectly via the idx array sinc this array contains all the indices of the data used 
            // for building the current node. Thus, sort this index wrt. to the current feature.

            // std::sort(XTmp.begin(), XTmp.end());
            unsigned int k = 0;
            for (auto j : idx) {
                XTmp[k].xf = X(j,i);
                XTmp[k].label = Y(j);
                ++k;
            }            

            std::sort(XTmp.begin(), XTmp.end(), [](const Sample & a, const Sample & b) -> bool { 
                return a.xf < b.xf; }
            );

            // Re-set class statistics
            std::fill(left_cnts.begin(), left_cnts.end(), 0);
            std::fill(right_cnts.begin(), right_cnts.end(), 0);
            
            bool first = true;
            unsigned int begin = 0; 
            internal_t best_threshold;

            // unsigned int jfirst = idx[0];
            for (unsigned int j = 0; j < n_data; ++j) {
                if (XTmp[j].xf == XTmp[0].xf) {
                    left_cnts[XTmp[j].label] += 1;
                } else {
                    if (first) {
                        best_threshold = XTmp[0].xf / 2.0 + XTmp[j].xf / 2.0;
                        //best_threshold = 0.5 * (f_values[0].first + f_values[j].first); 
                        first = false;
                        begin = j;
                    }
                    right_cnts[XTmp[j].label] += 1;
                }
            }

            if (first) {
                // We never choose a threshold which means that X[idx[0]][i] = X[idx[1]][i] = ... = X[idx[end]][i]. 
                // This will not give us a good split, so ignore this feature
                feature_is_const[i] = true;
                continue;
            }
            // Compute the corresponding gini score 
            internal_t best_score;
            if constexpr(tree_init == DT::INIT::GINI) {
                best_score = gini(left_cnts, right_cnts);
            } else {
                best_score = (*score)(left_cnts, right_cnts);
            }

            // Repeat what we have done above with the initial scanning, but now update left_cnts / right_cnts appropriately.
            unsigned int j = begin;

            while(j < n_data) {
                auto lj = j;
                do {
                    left_cnts[XTmp[j].label] += 1;
                    right_cnts[XTmp[j].label] -= 1;
                    ++j;
                } while(j < n_data && XTmp[j].xf == XTmp[lj].xf);
                
                if (j >= n_data) break;
 
                // internal_t cur_gini = gini(left_cnts, right_cnts);
                internal_t cur_score;
                if constexpr(tree_init == DT::INIT::GINI) {
                    cur_score = gini(left_cnts, right_cnts);
                } else {
                    cur_score = (*score)(left_cnts, right_cnts);
                }
                data_t threshold = XTmp[lj].xf / 2.0 + XTmp[j].xf / 2.0;
                if (cur_score < best_score) {
                    best_score = cur_score;
                    best_threshold = threshold;
                    //best_threshold = 0.5 * (f_values[j].first + f_values[j + 1].first);
                }
            }

            // Check if we not have already select a split or if this split is better than the other splits we found so far.
            // If so, then set this split
            if (!split_set || best_score < overall_best_score) {
                overall_best_score = best_score;
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
            return std::optional<std::pair<internal_t, unsigned int>>{std::make_pair(overall_best_threshold, overall_best_feature)};
        }
    }

    void make_leaf(int pid, std::vector<internal_t> &preds, bool is_left) {
        // Normalize the leaf predictions to be proper probabilities (sum to 1)
        internal_t sum = std::accumulate(preds.begin(), preds.end(), internal_t(0.0));
        if (sum > 0) {
            std::transform(preds.begin(), preds.end(), preds.begin(), [sum](auto& c){return 1.0/sum*c;});
        } else {
            std::fill_n(preds.begin(), n_classes, internal_t(1.0 / n_classes));
        }

        if (pid >= 0) {
            auto & parent = _nodes[pid];
            if (is_left) {
                parent.left_is_leaf = true;
                parent.left = _leaves.size();
            } else {
                parent.right_is_leaf = true;
                parent.right = _leaves.size();
            }
        }
        _leaves.insert(_leaves.end(), preds.begin(), preds.end());
    }

public:
    DecisionTree(unsigned int n_classes, unsigned int max_depth, unsigned int max_features, unsigned long seed) : n_classes(n_classes),max_depth(max_depth),max_features(max_features),seed(seed),score(std::nullopt) {
        static_assert(tree_init == DT::INIT::GINI || tree_init == DT::INIT::RANDOM, "You used DT::INIT::CUSTOM, but did not supply a score function. Please use another constructor and supply the score function.");
    }

    DecisionTree(unsigned int n_classes, unsigned int max_depth, unsigned int max_features, unsigned long seed, std::function< internal_t(std::vector<unsigned int> const &, std::vector<unsigned int> const &)> score) : n_classes(n_classes),max_depth(max_depth),max_features(max_features),seed(seed),score(score)  {
        static_assert(tree_init == DT::INIT::CUSTOM, "You used DT::INIT::GINI or DT::INIT::SCORE, but also supplied a score function. Please use another constructor that does not require a score function.");
    }

    unsigned int num_bytes() const {
        unsigned int node_size = 0;
        
        for (auto const &n : _nodes) {
            node_size += n.num_bytes();
        }

        return sizeof(*this) + node_size + sizeof(internal_t) * _leaves.size();
    }

    inline unsigned int leaf_index(matrix1d<data_t> const &x) const {
        unsigned int idx = 0;

        // On small datasets / batchs there might be no node fitted. In this case we only have leaf nodes
        if (_nodes.size() > 0) {
            while(true){
                auto const & n = _nodes[idx];
                if (x(n.idx) <= n.threshold) {
                    idx = _nodes[idx].left;
                    if (n.left_is_leaf) break;
                } else {
                    idx = _nodes[idx].right;
                    if (n.right_is_leaf) break;
                }
            }
        } 
        return idx;
    }

    std::vector<internal_t> & leaves() {
        return _leaves;
    };
    
    std::vector<Node<data_t>> & nodes() {
        return _nodes;
    };

    Tree<data_t>* clone(unsigned int seed) const {
        if constexpr(tree_init == DT::INIT::CUSTOM) {
            return new DecisionTree<data_t, tree_init>(n_classes, max_depth, max_features, seed, *score);
        } else {
            return new DecisionTree<data_t, tree_init>(n_classes, max_depth, max_features, seed);
        }
    };

    void predict_proba(matrix2d<data_t> const &X, matrix2d<internal_t> & preds) {
        for (unsigned int i = 0; i < X.rows; ++i) {
            internal_t const * const node_preds = &_leaves[leaf_index(X(i))]; //.preds.get();
            std::copy(node_preds, node_preds+n_classes, preds(i).begin());
        }
    }

    matrix2d<internal_t> predict_proba(matrix2d<data_t> const &X) {
        matrix2d<internal_t> preds(X.rows, n_classes);

        for (unsigned int i = 0; i < X.rows; ++i) {
            internal_t const * const node_preds = &_leaves[leaf_index(X(i))]; //.preds.get();
            std::copy(node_preds, node_preds+n_classes, preds(i).begin());
        }
        return preds;
    }

    unsigned int num_nodes() const {
        return _nodes.size() + int(_leaves.size() / n_classes);
    }

    // void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y) {
    //     matrix1d<unsigned int> idx(X.rows);
    //     std::iota(idx.begin(), idx.end(), 0);

    //     std::vector<bool> feature_is_const(X.cols, false);

    //     this->fit(X,Y,idx,feature_is_const);
    // }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, std::optional<std::reference_wrapper<const matrix1d<unsigned int>>> idx = std::nullopt) {
        std::vector<bool> feature_is_const(X.cols, false);

        if (idx.has_value()) {
            const matrix1d<unsigned int>& idx_ref = *idx;
            this->fit(X,Y,idx_ref,feature_is_const);
        } else {
            matrix1d<unsigned int> idx_ref(X.rows);
            std::iota(idx_ref.begin(), idx_ref.end(), 0);
            this->fit(X,Y,idx_ref,feature_is_const);
        }
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, matrix1d<unsigned int> const & idx, std::vector<bool> & feature_is_const) {
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
            std::vector<bool> feature_is_const;
            int parent;
            bool left;
            unsigned int depth;

            // I want to make sure that these objects are only moved and never copied. I expect the code below to not 
            // use any copy c'tors, but for safe measures we delete the copy constructor entirely.
            TreeExpansion(const TreeExpansion&) = delete;
            TreeExpansion() = default;
            TreeExpansion(TreeExpansion &&) = default;
        };
        if (max_features == 0) max_features = X.cols;

        //std::cout << "Moving a vector of size " << feature_is_const.size() << std::endl;

        std::queue<TreeExpansion> to_expand; 
        TreeExpansion root;
        // root.idx = std::move(idx);
        root.idx = std::vector<unsigned int>(idx.dim);
        std::copy(idx.begin(), idx.end(), root.idx.begin());

        root.parent = -1;
        root.left = false;
        root.depth = 0;
        root.feature_is_const = std::move(feature_is_const);
        //std::cout << "Checking a vector of size " << root.feature_is_const.size() << std::endl;

        to_expand.push(std::move(root));

        std::mt19937 gen(seed);

        // TODO: Maybe reserve some size in nodes?
        while(to_expand.size() > 0) {
            unsigned int cur_idx = _nodes.size();
            auto exp = std::move(to_expand.front());
            to_expand.pop();

            std::vector<internal_t> preds(n_classes, 0.0);
            for (auto i : exp.idx) {
                preds[Y(i)]++;
            }

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
                std::optional<std::pair<internal_t, unsigned int>> split;
                if constexpr (tree_init == DT::INIT::GINI || tree_init == DT::INIT::CUSTOM) {
                    split = best_split(X, Y, exp.idx, n_classes, max_features, gen, exp.feature_is_const);
                } else {
                    split = random_split(X, Y, exp.idx, gen, exp.feature_is_const);
                }

                if (split.has_value()) {
                    // A suitable split as been found
                    // To mitigate costly copy operations of the entire node, we first add it do the vector and then work
                    // on the reference via nodes[cur_idx]
                    _nodes.push_back(Node<data_t>());
                    if (exp.parent >= 0) {
                        if (exp.left) {
                            _nodes[exp.parent].left = cur_idx;
                            _nodes[exp.parent].left_is_leaf = false;
                        } else {
                            _nodes[exp.parent].right = cur_idx;
                            _nodes[exp.parent].right_is_leaf = false;
                        }
                    }

                    auto t = split.value().first;
                    auto f = split.value().second;
                    _nodes[cur_idx].idx = f;
                    _nodes[cur_idx].threshold = t;

                    // We do not need to store the predictions in inner nodes. Thus delete them here
                    // If we want to perform post-pruning at some point we should probably keep these statistics
                    //nodes[cur_idx].preds.reset();

                    TreeExpansion exp_left;
                    exp_left.parent = cur_idx;
                    exp_left.left = true;
                    exp_left.depth = exp.depth + 1;
                    exp_left.feature_is_const = exp.feature_is_const;

                    TreeExpansion exp_right;
                    exp_right.parent = cur_idx;
                    exp_right.left = false;
                    exp_right.depth = exp.depth + 1;
                    exp_right.feature_is_const = exp.feature_is_const;

                    // Split the data and expand the tree construction
                    for (auto i : exp.idx) {
                        if (X(i,f) <= t) {
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

    void load(matrix1d<internal_t> const & nodes) {
        unsigned int n_nodes = nodes(0);
        unsigned int n_leaves = nodes(1);

        _nodes = std::vector<Node<data_t>>(n_nodes);

        unsigned int j = 2;
        for (unsigned int i = 0; i < n_nodes; ++i, j += 6) {
            Node<data_t> &n = _nodes[i];
            // Node n;
            n.threshold = static_cast<data_t>(nodes(j));
            n.idx = static_cast<unsigned int>(nodes(j+1));
            n.left = static_cast<unsigned int>(nodes(j+2));
            n.right = static_cast<unsigned int>(nodes(j+3));
            n.left_is_leaf = nodes(j+4) == 0.0 ? false : true;
            n.right_is_leaf = nodes(j+5) == 0.0 ? false : true;
        }

        _leaves = std::vector<internal_t>(n_leaves);
        std::copy(&nodes.data[j], &nodes.data[j+n_leaves*n_classes], _leaves.begin());
        // _leafs = std::move(t_leafs);
        // _leafs = std::move(new_leafs);
    }

    matrix1d<internal_t> store() const {
        matrix1d<internal_t> nodes(2 + 6 * _nodes.size() + n_classes * _leaves.size());
        nodes(0) = _nodes.size();
        nodes(1) = _leaves.size();

        unsigned int i = 2;
        for (auto const &n : _nodes) {
            nodes(i++) = static_cast<internal_t>(n.threshold);
            nodes(i++) = static_cast<internal_t>(n.idx);
            nodes(i++) = static_cast<internal_t>(n.left);
            nodes(i++) = static_cast<internal_t>(n.right);
            nodes(i++) = static_cast<internal_t>(n.left_is_leaf);
            nodes(i++) = static_cast<internal_t>(n.right_is_leaf);
        }
        
        std::copy(_leaves.begin(), _leaves.end(), &nodes.data[i]);
        return nodes;
    }
};