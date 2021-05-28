#ifndef TREE_H
#define TREE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <queue>
#include <optional>
#include <string_view>

#include "Datatypes.h"

enum TREE_INIT {TRAIN, RANDOM};
enum TREE_NEXT {GRADIENT, NONE, INCREMENTAL};

template <typename pred_t>
class Node {
public:
    data_t threshold;
    unsigned int feature;
    unsigned int left, right;
    std::vector<pred_t> preds;
    // pred_t * preds;

    unsigned int num_bytes() const {
        return sizeof(data_t) + 3*sizeof(unsigned int) + sizeof(pred_t) * preds.size() + sizeof(std::vector<pred_t>);
    }

    Node(data_t threshold, unsigned int feature) : threshold(threshold), feature(feature) {}
    Node() = default;
    
    // ~Node() {
    //     delete[] preds;
    // }
};

template <TREE_INIT tree_init, TREE_NEXT tree_next, typename pred_t>
class Tree {
private:
    std::vector<Node<pred_t>> nodes;
    unsigned int n_classes;
    std::mt19937 gen; 

    inline unsigned int node_index(std::vector<data_t> const &x) const {
        unsigned int idx = 0;

        while(nodes[idx].left != 0) { /* or nodes[idx].right != 0 */
            auto const f = nodes[idx].feature;
            if (x[f] <= nodes[idx].threshold) {
                idx = nodes[idx].left;
            } else {
                idx = nodes[idx].right;
            }
        }
        return idx;
    }

    // static auto random_node(std::vector<bool> const &is_nominal, std::mt19937 &gen) {
    //     std::uniform_int_distribution<> idis(0, is_nominal.size() - 1);
    //     std::uniform_real_distribution<> fdis(0,1);
        
    //     unsigned int feature = idis(gen);
    //     data_t threshold;
    //     if (is_nominal[feature]) {
    //         threshold = 0.5; 
    //     } else {
    //         threshold = fdis(gen);
    //     }

    //     return std::pair<data_t, unsigned int>(threshold, feature);
    // }

     /**
     * @brief  Compute a random split for the given data. This algorithm has O(d * log d + d * N) runtime in the worst case, but should usually run in O(d * log d + N), where N is the number of examples and d is the number of features.
     * This implementation ensures that the returned split splits the data so that each child is non-empty if applied to the given data (at-least one example form X is routed towards left and towards right). The only exception occurs if X is empty or contains one example. In this case we return feature 0 with threshold 1. Threshold-values are placed in the middle between two samples. 
     * @note   
     * @param  &X: The example-set which is used to compute the splitting
     * @param  &Y: The label-set which is used to compute the splitting
     * @param  n_classes: The number of classes
     * @retval The best split as a std::pair<data_t, unsigned int>(best_threshold, best_feature) where the first entry is the threshold and the second entry the feature index.
     */
    static std::optional<std::pair<data_t, unsigned int>> random_split(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, std::mt19937 &gen) {
        // if (X.size() <= 1) {
        //     return random_node(is_nominal, gen);
        // }

        // We want to split at a random feature. However, we also want to ensure that the left / right child receive at-least one example with this random
        // split. Sometimes there are features which cannot ensure this (e.g. a binary features are '1'). Thus, we iterate over a random permutation of features 
        // and return as soon as we find a valid split
        std::vector<unsigned int> features(X[0].size());
        std::iota(std::begin(features), std::end(features), 0); 
        std::shuffle(features.begin(), features.end(), gen);

        for (auto const & f: features) {
            // We need to find the next smallest and next biggest value of the data to ensure that left/right will receive at-least 
            // one example. This is a brute force implementation in O(N)
            data_t smallest, second_smallest;
            if(X[0][f] <X[1][f]){
                smallest = X[0][f];
                second_smallest = X[1][f];
            } else {
                smallest = X[1][f];
                second_smallest = X[0][f];
            }

            data_t biggest, second_biggest;
            if(X[0][f] > X[1][f]){
                biggest = X[0][f];
                second_biggest = X[1][f];
            } else {
                biggest = X[1][f];
                second_biggest = X[0][f];
            }

            for (unsigned int i = 2; i < X.size(); ++i) {
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
            //return std::make_pair<data_t, unsigned int>(static_cast<data_t>(fdis(gen)), ftmp);
        }

        return std::nullopt;
        //return random_node(is_nominal, gen);
        //return std::make_pair<data_t, unsigned int>(1.0, 0);
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
     * TODO: Change code for tie-breaking
     * @note   
     * @param  &X: The example-set which is used to compute the splitting
     * @param  &Y: The label-set which is used to compute the splitting
     * @param  n_classes: The number of classes
     * @retval The best split as a std::pair<data_t, unsigned int>(best_threshold, best_feature) where the first entry is the threshold and the second entry the feature index.
     */
    static std::optional<std::pair<data_t, unsigned int>> best_split(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, long n_classes, std::mt19937 &gen) {
        // if (X.size() <= 1) {
        //     return std::make_pair(1.0, static_cast<unsigned int>(0));
        //     //return random_node(is_nominal, gen);
        // }

        unsigned int n_data = X.size();
        unsigned int n_features = X[0].size();

        data_t overall_best_gini = 0;
        unsigned int overall_best_feature = 0;
        data_t overall_best_threshold = 0;
        bool split_set = false;
        std::vector<unsigned int> features(n_features);
        std::iota(features.begin(),features.end(), 0); 
        std::shuffle(features.begin(), features.end(), gen);

        for (auto i: features) {
            // In order to compute the best spliting threshold for the current feature we need to evaluate every possible split value.
            // These can be up to n_data - 1 points and for each threshold we need to evaluate if they belong to the left or right child. 
            // The naive implementation thus require O(n_data**2) runtime. We use a slightly more optimized version which requires O(n_data * log n_data). 
            // To do so, we first the examples according to their feature values and compute the initial statistics for the left/right child. Then, we gradually 
            // move the split-threshold to the next value and onyl update the statistics.

            // Copy feature values and targets into new vector
            std::vector<std::pair<data_t, unsigned int>> f_values(n_data);
            for (unsigned int j = 0; j < n_data; ++j) {
                f_values[j] = std::make_pair(X[j][i], Y[j]);
            }
            // By default sort sorts after the first feature
            std::sort(f_values.begin(), f_values.end());
            //data_t max_t = f_values[n_data - 1].first;

            // Prepare class statistics
            std::vector<unsigned int> left_cnts(n_classes);
            std::vector<unsigned int> right_cnts(n_classes);
            std::fill(left_cnts.begin(), left_cnts.end(), 0);
            std::fill(right_cnts.begin(), right_cnts.end(), 0);
            
            bool first = true;
            unsigned int begin = 0; 
            data_t best_threshold;

            for (unsigned int j = 0; j < n_data; ++j) {
                if (f_values[j].first == f_values[0].first) {
                    left_cnts[f_values[j].second] += 1;
                } else {
                    if (first) {
                        best_threshold = f_values[0].first / 2.0 + f_values[j].first / 2.0;
                        //best_threshold = 0.5 * (f_values[0].first + f_values[j].first); 
                        first = false;
                        begin = j;
                    }
                    right_cnts[f_values[j].second] += 1;
                }
            }
            
            if (first) {
                // We never choose a threshold which means that f_values[0] = f_values[1] = ... = f_values[end]. 
                // This will not give us a good split, so ignore this feature
                continue;
            }
            // Compute the corresponding gini score 
            data_t best_gini = gini(left_cnts, right_cnts);

            // Repeat what we have done above with the initial scanning, but now update left_cnts / right_cnts appropriately.
            unsigned int j = begin;

            while(j < n_data) {
                data_t left = f_values[j].first;

                do {
                    auto const & f = f_values[j];
                    left_cnts[f.second] += 1;
                    right_cnts[f.second] -= 1;
                    ++j;
                } while(j < n_data && f_values[j].first == left);
                
                if (j >= n_data) break;

                data_t cur_gini = gini(left_cnts, right_cnts);
                data_t threshold = left / 2.0 + f_values[j].first / 2.0;
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
        }

        if (!split_set) {
            return std::nullopt;
        } else {
            return std::optional<std::pair<data_t, unsigned int>>{std::make_pair(overall_best_threshold, overall_best_feature)};
        }
    }

    static auto insert_leaf(std::vector<pred_t> &class_cnt, unsigned int n_classes) {
        Node<pred_t> cur_node;
        cur_node.left = 0;
        cur_node.right = 0;
        cur_node.preds = class_cnt;

        if constexpr (tree_next != INCREMENTAL) {
            data_t sum = std::accumulate(class_cnt.begin(), class_cnt.end(), pred_t(0.0));
            if (sum > 0) {
                std::transform(cur_node.preds.begin(), cur_node.preds.end(), cur_node.preds.begin(), [sum](auto& c){return 1.0/sum*c;});
            } else {
                std::fill(cur_node.preds.begin(), cur_node.preds.end(), 1.0/n_classes);
            }
        }
        return cur_node;
    }

    void train(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int max_depth) {
        struct TreeExpansion {
            std::vector<std::vector<data_t>> x;
            std::vector<unsigned int> y;
            int parent;
            bool left;
            unsigned int depth;
            // TreeExpansion(std::vector<std::vector<data_t>> const &x, std::vector<unsigned int> const &y, int parent, bool left, unsigned int depth) 
            //     : x(x),y(y),parent(parent),left(left),depth(depth) {}
        };

        std::queue<TreeExpansion> to_expand; 
        //to_expand.push(TreeExpansion(X, Y, -1, false, 0));
        TreeExpansion root;
        root.x = X;
        root.y = Y;
        root.parent = -1;
        root.left = false;
        root.depth = 0;
        to_expand.push(root);

        while(to_expand.size() > 0) {
            unsigned int cur_idx = nodes.size();
            auto exp = to_expand.front();
            to_expand.pop();

            std::vector<pred_t> class_cnt(n_classes, 0.0);
            for (unsigned int i = 0; i < exp.x.size(); ++i) {
                class_cnt[exp.y[i]]++;
            }

            bool is_leaf = false;
            for (unsigned int i = 0; i < n_classes; ++i) {
                if (class_cnt[i] == exp.y.size()) {
                    is_leaf = true;
                    break;
                }
            }

            if (is_leaf || exp.depth >= max_depth) {
                auto cur_node = insert_leaf(class_cnt, n_classes);

                nodes.push_back(cur_node);
                if (exp.parent >= 0) {
                    if (exp.left) {
                        nodes[exp.parent].left = cur_idx;
                    } else {
                        nodes[exp.parent].right = cur_idx;
                    }
                }
            } else {
                std::optional<std::pair<data_t, unsigned int>> split;
                if constexpr (tree_init == TRAIN) {
                    split = best_split(exp.x, exp.y, n_classes, gen);
                } else {
                    split = random_split(exp.x, exp.y, gen);
                }

                if (split.has_value()) {
                    auto t = split.value().first;
                    auto f = split.value().second;
                    nodes.push_back(Node<pred_t>(t,f));
                    
                    if (exp.parent >= 0) {
                        if (exp.left) {
                            nodes[exp.parent].left = cur_idx;
                        } else {
                            nodes[exp.parent].right = cur_idx;
                        }
                    }

                    TreeExpansion exp_left;
                    exp_left.parent = cur_idx;
                    exp_left.left = true;
                    exp_left.depth = exp.depth + 1;

                    TreeExpansion exp_right;
                    exp_right.parent = cur_idx;
                    exp_right.left = false;
                    exp_right.depth = exp.depth + 1;

                    for (unsigned int i = 0; i < exp.x.size(); ++i) {
                        if (exp.x[i][f] <= t) {
                            exp_left.x.push_back(exp.x[i]);
                            exp_left.y.push_back(exp.y[i]);
                        } else {
                            exp_right.x.push_back(exp.x[i]);
                            exp_right.y.push_back(exp.y[i]);
                        }
                    }

                    to_expand.push(exp_left);
                    to_expand.push(exp_right);

                    // std::vector<std::vector<data_t>> XLeft, XRight;
                    // std::vector<unsigned int> YLeft, YRight;

                    // XLeft.reserve(exp.x.size());
                    // XRight.reserve(exp.x.size());
                    // YLeft.reserve(exp.x.size());
                    // YRight.reserve(exp.x.size());
                    // for (unsigned int i = 0; i < exp.x.size(); ++i) {
                    //     if (exp.x[i][f] <= t) {
                    //         XLeft.push_back(exp.x[i]);
                    //         YLeft.push_back(exp.y[i]);
                    //     } else {
                    //         XRight.push_back(exp.x[i]);
                    //         YRight.push_back(exp.y[i]);
                    //     }
                    // }

                    // to_expand.push(TreeExpansion(XLeft, YLeft, cur_idx, true, exp.depth+1));
                    // to_expand.push(TreeExpansion(XRight, YRight, cur_idx, false, exp.depth+1));
                } else {
                    auto cur_node = insert_leaf(class_cnt, n_classes);

                    nodes.push_back(cur_node);
                    if (exp.parent >= 0) {
                        if (exp.left) {
                            nodes[exp.parent].left = cur_idx;
                        } else {
                            nodes[exp.parent].right = cur_idx;
                        }
                    }
                }

            }
        }
    }

public:

    Tree(unsigned int max_depth, unsigned int n_classes, unsigned long seed, std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) : n_classes(n_classes), gen(seed) {
        train(X, Y, max_depth);
    }

    unsigned int num_bytes() const {
        unsigned int node_size = 0;
        
        for (auto const &n : nodes) {
            node_size += n.num_bytes();
        }

        return 3 * sizeof(unsigned int) + node_size + sizeof(std::mt19937);
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, std::vector<std::vector<data_t>> const &tree_grad, data_t step_size) {
        if constexpr (tree_next == GRADIENT) {
            for (unsigned int i = 0; i < X.size(); ++i) {
                auto idx = node_index(X[i]);
                for (unsigned int j = 0; j < n_classes; ++j) {
                    nodes[idx].preds[j] = nodes[idx].preds[j] - step_size * tree_grad[i][j];
                } 
            }
        } else if constexpr (tree_next == INCREMENTAL) {
            for (unsigned int i = 0; i < X.size(); ++i) {
                auto idx = node_index(X[i]);
                nodes[idx].preds[Y[i]]++;
            }
        } else {
            /* Do nothing */
        }
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> preds(X.size());
        for (unsigned int i = 0; i < X.size(); ++i) {
            preds[i] = nodes[node_index(X[i])].preds;
        }
        return preds;
    }

    unsigned int get_num_nodes() const {
        return nodes.size();
    }
};

#endif