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
#include <zlib.h>
#include <string.h>
#include <cstring>

#include "Optimizer.h"
#include "Tree.h"

namespace DDT {
    enum TREE_INIT {TRAIN, RANDOM};
    enum DISTANCE {EUCLIDEAN, ZLIB};
}

unsigned int compress_len(char const * const data, unsigned int n_bytes, int level = Z_BEST_SPEED) {
    // Use zlib's compressBound function to calculate the maximum size of the compressed data buffer
    auto n_bytes_compressed = compressBound(n_bytes);
    char outbuffer[n_bytes_compressed];

    // Perform the compression
    int result = compress2((Bytef*)outbuffer, &n_bytes_compressed, (const Bytef*)data, n_bytes, level);

    if (result != Z_OK) {
        throw(std::runtime_error("Errorung during zlib compression. Error code was " + std::to_string(result)));
    }

    return n_bytes_compressed;
}

template <DDT::TREE_INIT tree_init, DDT::DISTANCE distance_type, OPTIMIZER::OPTIMIZER_TYPE tree_opt>
class DistanceDecisionTree { //: public Tree {

// template <LOSS::TYPE friend_loss_type, OPTIMIZER::OPTIMIZER_TYPE friend_opt, OPTIMIZER::OPTIMIZER_TYPE friend_tree_opt, DDT::TREE_INIT friend_tree_init>
// friend class ShrubEnsemble;

private:
    std::vector<Node> _nodes;
    matrix2d<data_t> examples; 
    std::vector<internal_t> _leafs;
    unsigned int n_classes;
    unsigned int max_depth;
    unsigned int max_examples;
    unsigned int lambda;
    unsigned long seed;

    char * tmp_concat_data; // TODO Make the existence conditioned on distance_type

    OPTIMIZER::Optimizer<tree_opt,OPTIMIZER::STEP_SIZE_TYPE::CONSTANT> optimizer;

    inline unsigned int leaf_index(matrix1d<data_t> const &x) const {
        unsigned int idx = 0;

        // On small datasets / batchs there might be no node fitted. In this case we only have leaf nodes
        if (_nodes.size() > 0) {
            while(true){
                auto const & n = _nodes[idx];
                auto const & ref = examples(n.idx);

                if (distance(ref,x) <= n.threshold) {
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

    internal_t distance(matrix1d<data_t> const &x1, matrix1d<data_t> const &x2) const {
        if constexpr (distance_type == DDT::DISTANCE::EUCLIDEAN) {
            return std::inner_product(x1.begin(), x1.end(), x2.begin(), data_t(0), 
                std::plus<data_t>(), [](data_t x,data_t y){return (y-x)*(y-x);}
            );
        } else {
            const char * d1 = reinterpret_cast<const char *>(x1.begin());
            unsigned int n1 = sizeof(data_t)*x1.dim;
            unsigned int len_n1 = compress_len(d1, n1);

            const char * d2 = reinterpret_cast<const char *>(x2.begin());
            unsigned int n2 = sizeof(data_t)*x2.dim;
            unsigned int len_n2 = compress_len(d2, n2);
            
            // TODO This can be a temporary field in the class which we create once per fit
            char * concat_data = new char[n1+n2];

            std::memcpy(concat_data, d1, n1);
            std::memcpy(concat_data + n1, d2, n2);

            unsigned int len_concat = compress_len(concat_data, n1+n2);

            delete[] concat_data;
            return static_cast<internal_t>(len_concat - std::min(len_n1, len_n2)) / static_cast<internal_t>(std::max(len_n1, len_n2));
        }
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
    std::optional<std::pair<internal_t, unsigned int>> random_split(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, std::vector<unsigned int> const & idx, std::mt19937 &gen) {
        // At-least 2 points are required for splitting.
        // Technically this check is unncessary since we stop tree construction once there is only one label in the data which is always the case 
        // if we have 0 or 1 examples. For safety measure we keep this check alive at the moment.
        if (idx.size() < 2) {
            return std::nullopt;
        }

        // We want to split at a random distance. However, we also want to ensure that the left / right child receive at-least one example with this random
        // split. Sometimes there are distances which cannot ensure this (e.g. wenn all distances are the same). Thus, we iterate over a random permutation of distances 
        // and return as soon as we find a valid split
        std::vector<unsigned int> ref_example(X.cols);
        std::iota(std::begin(ref_example), std::end(ref_example), 0); 
        std::shuffle(ref_example.begin(), ref_example.end(), gen);
 
        for (auto const & cur_idx: ref_example) {
            std::vector<internal_t> distances(idx.size());
            for (unsigned int i = 0; i < idx.size(); ++i) {
                distances[i] = distance(X(cur_idx), X(idx[i]));
            }

            std::sort(distances.begin(), distances.end());
            unsigned int isecondmax = distances.size();
            for (int i = distances.size(); i >= 0; --i) {
                if (distances[i] < distances[isecondmax]) {
                    isecondmax = i;
                    break;
                }
            }
            if (isecondmax == distances.size()) continue;

            std::uniform_real_distribution<> fdis(distances[0], distances[isecondmax]); 

            // So usually I would expect the following line to work, but for some reason it does not. Is this a gcc bug?
            //return std::make_pair<data_t, unsigned int>(static_cast<data_t>(fdis(gen)), f);
            int idx_tmp = cur_idx;
            return std::optional<std::pair<internal_t, unsigned int>>{std::make_pair<internal_t, unsigned int>(static_cast<data_t>(fdis(gen)), idx_tmp)};
        }

        return std::nullopt;
    }

    static internal_t score(internal_t left_m, internal_t left_var, std::vector<unsigned int> const &left_cnts, internal_t right_m, internal_t right_var, std::vector<unsigned int> const &right_cnts, internal_t lambda) {
        return lambda * (left_var / 2.0 + right_var / 2.0) + (1.0 - lambda) * gini(left_cnts, right_cnts);
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
    std::optional<std::pair<internal_t, unsigned int>> best_split(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, std::vector<unsigned int> &idx, matrix2d<internal_t> const & distance_matrix, long n_classes, unsigned int max_examples, std::mt19937 &gen, internal_t lambda) {
        // At-least 2 points are required for splitting.
        // Technically this check is unncessary since we stop tree construction once there is only one label in the data which is always the case 
        // if we have 0 or 1 examples. For safea measure we keep this check alive however.
        if (idx.size() < 2) {
            return std::nullopt;
        }

        unsigned int n_data = idx.size();

        data_t overall_best_score = 0;
        unsigned int overall_best_example = 0;
        data_t overall_best_threshold = 0;
        bool split_set = false;

        // Sometimes multiple examples have equally good splits (= same score). Thus, 
        // we tierate over the examples in a random order to promote some diversity. 
        std::shuffle(idx.begin(), idx.end(), gen);

        // Prepare class statistics
        std::vector<unsigned int> left_cnts(n_classes);
        std::vector<unsigned int> right_cnts(n_classes);

        unsigned int left_n = 0, right_n = 0;
        internal_t left_ss = 0, right_ss = 0;
        internal_t left_m = 0, right_m = 0;

        unsigned int ecnt = 0;
        for (auto i: idx) {
            // In order to compute the best spliting threshold for the current feature we need to evaluate every possible split value.
            // These can be up to n_data - 1 points and for each threshold we need to evaluate if they belong to the left or right child. 
            // The naive implementation thus require O(n_data^2) runtime. We use a slightly more optimized version which requires O(n_data * log n_data). 
            // To do so, we first the examples according to their feature values and compute the initial statistics for the left/right child. Then, we gradually 
            // move the split-threshold to the next value and onyl update the statistics.

            // The data is always accessed indirectly via the idx array since this array contains all the indices of the data used 
            // for building the current node. Thus, make sure distances[] contains the appropriate distances for the current ref point (i) wrt. to 
            // all training data idx[j]

            // For larger datasets this is probaly more memory friendly, but it is also quite slow
            // std::vector<internal_t> distances(idx.size());
            // for (unsigned int j = 0; j < idx.size(); ++j) {
            //     distances[j] = distance(X(i), X(idx[j]));
            // }

            std::vector<internal_t> distances(idx.size());
            for (unsigned int j = 0; j < idx.size(); ++j) {
                distances[j] = distance_matrix(i, idx[j]);
            }

            std::sort(distances.begin(), distances.end());

            // Re-set class statistics
            std::fill(left_cnts.begin(), left_cnts.end(), 0);
            std::fill(right_cnts.begin(), right_cnts.end(), 0);
            
            bool first = true;
            unsigned int begin = 0; 
            data_t best_threshold;

            // unsigned int jfirst = idx[0];
            for (unsigned int j = 0; j < distances.size(); ++j) {
                auto d = distances[j]; 
                if (d == distances[0]) {
                    left_n++;
                    // The distances are the same so there is no variance and there the mean stays the same
                    left_ss = 0;
                    left_m = d;
                    // internal_t delta = d - left_m;
                    // left_m += delta / left_n;
                    // internal_t delta2 = d - left_m;
                    // left_ss += delta * delta2;                
                    left_cnts[Y(idx[j])] += 1;
                } else {
                    if (first) {
                        best_threshold = distances[0] / 2.0 + distances[j] / 2.0;
                        first = false;
                        begin = j;
                    }
                    // Update mean/variance on the fly via welfords algorithm
                    right_n++;
                    internal_t delta = d - right_m;
                    right_m += delta / right_n;
                    internal_t delta2 = d - right_m;
                    right_ss += delta * delta2;    
                    right_cnts[Y(idx[j])] += 1;
                }
            }

            if (first) {
                // We never choose a threshold which means that distance[0] == distance[1] = ... = distance[end]
                // This will not give us a good split, regardless of the reference point we use for computation. Hence, we may break out of this loop and return.
                break;
            }

            // Compute the corresponding score 
            internal_t left_var = left_ss == 0 || left_n <= 1 ? 0 : std::sqrt(left_ss / (left_n - 1));
            internal_t right_var = right_ss == 0 || right_n <= 1 ? 0 : std::sqrt(right_ss / (right_n - 1));
            data_t best_score = score(left_m, left_var, left_cnts, right_m, right_var, right_cnts, lambda);

            // Repeat what we have done above with the initial scanning, but now update left_cnts / right_cnts etc. appropriately.
            unsigned int j = begin;

            while(j < n_data) {
                auto lj = j;
                auto d = distances[j];
                do {
                    left_cnts[Y(idx[j])] += 1;
                    // Update mean/variance on the fly via welfords algorithm
                    left_n++;
                    internal_t delta = d - left_m;
                    left_m += delta / left_n;
                    internal_t delta2 = d - left_m;
                    left_ss += delta * delta2; 

                    // Execute Welfords algorithm backwards, i.e. remove distance from mean/variance computation
                    right_cnts[Y(idx[j])] -= 1;
                    delta = right_m;
                    right_m = right_m - (d - right_m) / (right_n - 1);
                    right_ss = right_ss - (d - right_m) * (d - delta);
                    right_n--;

                    ++j;
                } while(j < n_data && distances[lj] == distances[j]);
                
                if (j >= n_data) break;

                left_var = left_ss == 0 || left_n <= 1 ? 0 : std::sqrt(left_ss / (left_n - 1));
                right_var = right_ss == 0 || right_n <= 1 ? 0 : std::sqrt(right_ss / (right_n - 1));

                data_t cur_score = score(left_m, left_var, left_cnts, right_m, right_var, right_cnts, lambda);
                data_t threshold = distances[lj] / 2.0 + distances[j] / 2.0;
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
                overall_best_example = i;
                overall_best_threshold = best_threshold;
                split_set = true;
            } 

            // Evaluate at most max_examples, but keep looking for splits if we haven found a valid one yet
            ecnt += 1;
            if (ecnt >= max_examples && split_set) break;
        }

        // std::cout << "Choosing " << overall_best_example << " with score " << overall_best_score << " and threshold " << overall_best_threshold << std::endl;

        if (!split_set) {
            return std::nullopt;
        } else {
            return std::optional<std::pair<internal_t, unsigned int>>{std::make_pair(overall_best_threshold, overall_best_example)};
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
            auto & parent = _nodes[pid];
            if (is_left) {
                parent.left_is_leaf = true;
                parent.left = _leafs.size();
            } else {
                parent.right_is_leaf = true;
                parent.right = _leafs.size();
            }
        }
        _leafs.insert(_leafs.end(), preds.begin(), preds.end());
    }

public:

    DistanceDecisionTree(unsigned int n_classes, unsigned int max_depth, unsigned int max_examples, unsigned long seed, internal_t lambda, internal_t step_size) : n_classes(n_classes),max_depth(max_depth),max_examples(max_examples), lambda(lambda),seed(seed),optimizer(step_size) {}

    unsigned int num_bytes() const {
        unsigned int node_size = 0;
        
        for (auto const &n : _nodes) {
            node_size += n.num_bytes();
        }

        return sizeof(*this) + node_size + sizeof(internal_t) * _leafs.size() + optimizer.num_bytes() + examples.num_bytes();
    }

    unsigned int num_ref_examples() const {
        return examples.rows;
    }

    void predict_proba(matrix2d<data_t> const &X, matrix2d<data_t> & preds) {
        for (unsigned int i = 0; i < X.rows; ++i) {
            internal_t const * const node_preds = &_leafs[leaf_index(X(i))]; //.preds.get();
            std::copy(node_preds, node_preds+n_classes, preds(i).begin());
        }
    }

    matrix2d<data_t> predict_proba(matrix2d<data_t> const &X) {
        matrix2d<data_t> preds(X.rows, n_classes);

        for (unsigned int i = 0; i < X.rows; ++i) {
            internal_t const * const node_preds = &_leafs[leaf_index(X(i))]; //.preds.get();
            std::copy(node_preds, node_preds+n_classes, preds(i).begin());
        }
        return preds;
    }

    unsigned int num_nodes() const {
        return _nodes.size() + int(_leafs.size() / n_classes);
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y) {
        matrix1d<unsigned int> idx(X.rows);
        std::iota(idx.begin(), idx.end(), 0);

        matrix2d<internal_t> distance_matrix(idx.dim, idx.dim);
        for (unsigned int i = 0; i < idx.dim; ++i) {
            for (unsigned int j = i; j < idx.dim; ++j) {
                // idx are 1...X.rows, so no indirect access is necessary here
                distance_matrix(i,j) = distance(X(i), X(j));
                if (i != j) distance_matrix(j,i) = distance_matrix(i,j);
            }
        }

        this->fit(X,Y,idx,distance_matrix);
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, matrix1d<unsigned int> const & idx, matrix2d<data_t> const &distance_matrix) {
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
        if (max_examples == 0) max_examples = idx.dim;

        if constexpr(distance_type == DDT::DISTANCE::ZLIB) tmp_concat_data = new char[2*sizeof(data_t)*X.cols];

        std::queue<TreeExpansion> to_expand; 
        TreeExpansion root;
        root.idx = std::vector<unsigned int>(idx.dim);
        std::copy(idx.begin(), idx.end(), root.idx.begin());

        root.parent = -1;
        root.left = false;
        root.depth = 0;

        to_expand.push(std::move(root));

        std::mt19937 gen(seed);

        std::vector<unsigned int> example_idx;
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
                std::optional<std::pair<data_t, unsigned int>> split;
                if constexpr (tree_init == DDT::TREE_INIT::TRAIN) {
                    split = best_split(X, Y, exp.idx, distance_matrix, n_classes, max_examples, gen, lambda);
                } else {
                    split = random_split(X, Y, exp.idx, gen);
                }

                if (split.has_value()) {
                    // A suitable split as been found
                    // To mitigate costly copy operations of the entire node, we first add it do the vector and then work
                    // on the reference via nodes[cur_idx]
                    _nodes.push_back(Node());
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
                    auto e = split.value().second;
                    _nodes[cur_idx].idx = example_idx.size();
                    _nodes[cur_idx].threshold = t;
                    example_idx.push_back(e);

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
                        auto const & ref = X(e);
                        auto const & x = X(i);
                        if (distance(ref,x) <= t) {
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

            examples = matrix2d<data_t>(example_idx.size(), X.cols);
            for (unsigned int i = 0; i < example_idx.size(); ++i) {
                for (unsigned int j = 0; j < X.cols; ++j) {
                    examples(i,j) = X(example_idx[i],j);
                }
            }
        }

        if constexpr(distance_type == DDT::DISTANCE::ZLIB) delete[] tmp_concat_data;
    }

    void load(matrix1d<internal_t> const &  new_nodes, matrix1d<internal_t> const & new_leafs) {
        // TODO ADD examples as well
        _nodes = std::vector<Node>(static_cast<unsigned int>(new_nodes.dim / 6));

        unsigned int nnodes = 0;
        for (unsigned int j = 0; j < new_nodes.dim; j += 6) {
            Node &n = _nodes[nnodes];
            // Node n;
            n.threshold = static_cast<data_t>(new_nodes(j));
            n.idx = static_cast<unsigned int>(new_nodes(j+1));
            n.left = static_cast<unsigned int>(new_nodes(j+2));
            n.right = static_cast<unsigned int>(new_nodes(j+3));
            n.left_is_leaf = new_nodes(j+4) == 0.0 ? false : true;
            n.right_is_leaf = new_nodes(j+5) == 0.0 ? false : true;
            // t_nodes.push_back(n);
            nnodes++;
        }

        _leafs = std::vector<internal_t>(new_leafs.dim);
        std::copy(new_leafs.begin(), new_leafs.end(), _leafs.begin());
        // _leafs = std::move(t_leafs);
        // _leafs = std::move(new_leafs);
    }

    std::tuple<matrix1d<internal_t>, matrix1d<internal_t>> store() const {
        matrix1d<internal_t> primitive_nodes(6 * _nodes.size());
        // TODO ADD examples as well

        unsigned int i = 0;
        for (auto const &n : _nodes) {
            primitive_nodes(i++) = static_cast<internal_t>(n.threshold);
            primitive_nodes(i++) = static_cast<internal_t>(n.idx);
            primitive_nodes(i++) = static_cast<internal_t>(n.left);
            primitive_nodes(i++) = static_cast<internal_t>(n.right);
            primitive_nodes(i++) = static_cast<internal_t>(n.left_is_leaf);
            primitive_nodes(i++) = static_cast<internal_t>(n.right_is_leaf);
        }

        matrix1d<internal_t> t_leafs(_leafs.size());
        std::copy(_leafs.begin(), _leafs.end(), t_leafs.begin());
        return std::make_tuple<matrix1d<internal_t>, matrix1d<internal_t>> (std::move(primitive_nodes), std::move(t_leafs));
    }
};

// class DistanceDecisionTreeClassifier {
// protected:
// 	Tree * tree = nullptr;

// public:

//     DistanceDecisionTreeClassifier(
//         unsigned int n_classes, 
//         unsigned int max_depth, 
//         unsigned int max_examples,
//         data_t lambda,
//         unsigned long seed, 
//         internal_t step_size,
//         const std::string tree_init_mode, 
//         const std::string tree_optimizer
//     ) { 

//         // Yeha this is ugly and there is probably clever way to do this with C++17/20, but this was quicker to code and it gets the job done.
//         // Also, lets be real here: There is only a limited chance more init/next modes are added without much refactoring of the whole project
//         if (tree_init_mode == "random" && tree_optimizer == "sgd") {
//             tree = new DistanceDecisionTree<DDT::TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::SGD>(n_classes,max_depth,max_examples,seed,lambda,step_size);
//         } else if (tree_init_mode == "random" && tree_optimizer == "adam") {
//             tree = new DistanceDecisionTree<DDT::TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::ADAM>(n_classes,max_depth,max_examples,seed,lambda,step_size);
//         } else if (tree_init_mode == "random" && tree_optimizer == "none") {
//             tree = new DistanceDecisionTree<DDT::TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::NONE>(n_classes,max_depth,max_examples,seed,lambda,step_size);
//         } else if (tree_init_mode == "train" && tree_optimizer == "sgd") {
//             tree = new DistanceDecisionTree<DDT::TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::SGD>(n_classes,max_depth,max_examples,seed,lambda,step_size);
//         } else if (tree_init_mode == "train" && tree_optimizer == "adam") {
//             tree = new DistanceDecisionTree<DDT::TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::ADAM>(n_classes,max_depth,max_examples,seed,lambda,step_size);
//         } else if (tree_init_mode == "train" && tree_optimizer == "none") {
//             tree = new DistanceDecisionTree<DDT::TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::NONE>(n_classes,max_depth,max_examples,seed,lambda,step_size);
//         } else {
//             throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the three  optimizers {none,sgd,adam} are supported for trees, but you provided a combination of " + tree_init_mode + " and " + tree_optimizer);
//         }
//     }

//     void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y) {
//         if (tree != nullptr) {
//             tree->fit(X, Y);
//         } else {
//             throw std::runtime_error("The internal object pointer in DecisionTree was null. This should now happen!");
//         }
//     }

//     matrix2d<data_t> predict_proba(matrix2d<data_t> const &X) {
//         if (tree != nullptr) {
//             return tree->predict_proba(X);
//         } else {
//             throw std::runtime_error("The internal object pointer in DecisionTree was null. This should now happen!");
//         }
//     }
    
//     ~DistanceDecisionTreeClassifier() {
//         if (tree != nullptr) {
//             delete tree;
//         }
//     }

//     unsigned int num_ref_examples() const {
//         if (tree != nullptr) {
//             return static_cast<DistanceDecisionTree*>(tree)->num_ref_examples();
//         } else {
//             throw std::runtime_error("The internal object pointer in DecisionTree was null. This should now happen!");
//         }
//     }

//     unsigned int num_bytes() const {
//         if (tree != nullptr) {
//             return tree->num_bytes();
//         } else {
//             throw std::runtime_error("The internal object pointer in DecisionTree was null. This should now happen!");
//         }
//     }

//     unsigned int num_nodes() const {
//         if (tree != nullptr) {
//             return tree->num_nodes();
//         } else {
//             throw std::runtime_error("The internal object pointer in DecisionTree was null. This should now happen!");
//         }
//     }

//     void load(matrix1d<internal_t> const & new_nodes, matrix1d<internal_t> const & new_leafs) {
//         if (tree != nullptr) {
//             return tree->load(new_nodes,new_leafs);
//         } else {
//             throw std::runtime_error("The internal object pointer in DecisionTree was null. This should now happen!");
//         }
//     }

//     std::tuple<matrix1d<internal_t>, matrix1d<internal_t>> store() const {
//         if (tree != nullptr) {
//             return tree->store();
//         } else {
//             throw std::runtime_error("The internal object pointer in DecisionTree was null. This should now happen!");
//         }
//     }
// };