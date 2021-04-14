#ifndef ENSEMBLE_REGULARIZER_H
#define ENSEMBLE_REGULARIZER_H

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "Datatypes.h"

enum class ENSEMBLE_REGULARIZER {NO,L0,L1,hard_L1};

std::vector<data_t> no_reg(std::vector<data_t> const &w, data_t scale) {
    return w;
}

std::vector<data_t> L0_reg(std::vector<data_t> const &w, data_t scale) {
    std::vector<data_t> tmp_w(w.size());
    data_t tmp = std::sqrt(2 * scale);
    for (unsigned int i = 0; i < w.size(); ++i) {
        if (std::abs(w[i]) < tmp) {
            tmp_w[i] = 0;
        } else {
            tmp_w[i] = w[i];
        }
    }

    return tmp_w;
}

std::vector<data_t> L1_reg(std::vector<data_t> const &w, data_t scale) {
    std::vector<data_t> tmp_w(w.size());
    for (unsigned int i = 0; i < w.size(); ++i) {
        data_t sign = w[i] > 0 ? 1 : -1;
        tmp_w[i] = sign * std::max(0.0, std::abs(w[i])  - scale);
    }

    return tmp_w;
}

// https://stackoverflow.com/questions/14902876/indices-of-the-k-largest-elements-in-an-unsorted-length-n-array
std::vector<unsigned int> top_k(std::vector<data_t> const &a, unsigned int K) {
    std::vector<unsigned int> top_idx;
    std::priority_queue< std::pair<data_t, unsigned int>, std::vector< std::pair<data_t, unsigned int> >, std::greater <std::pair<data_t, unsigned int> > > q;
  
    for (unsigned int i = 0; i < a.size(); ++i) {
        if (q.size() < K) {
            q.push(std::pair<data_t, unsigned int>(a[i], i));
        } else if (q.top().first < a[i]) {
            q.pop();
            q.push(std::pair<data_t, unsigned int>(a[i], i));
        }
    }

    while (!q.empty()) {
        top_idx.push_back(q.top().second);
        q.pop();
    }

    return top_idx;
}

std::vector<data_t> hard_L1_reg(std::vector<data_t> const &w, data_t K) {
    std::vector<unsigned int> top_idx = top_k(w, K);
    std::vector<data_t> tmp_w(w.size(), 0);

    for (auto i : top_idx) {
        tmp_w[i] = w[i];
    }

    return tmp_w;
}

/*
* if x is None or len(x) == 0:
        return x
    sorted_x = np.sort(x)
    x_sum = sorted_x[0]
    l = 1.0 - sorted_x[0]
    for i in range(1,len(sorted_x)):
        x_sum += sorted_x[i]
        tmp = 1.0 / (i + 1.0) * (1.0 - x_sum)
        if (sorted_x[i] + tmp) > 0:
            l = tmp 
    
    return [max(xi + l, 0.0) for xi in x]
*/
std::vector<data_t> to_prob_simplex(std::vector<data_t> const &w) {
    if (w.size() == 0) {
        return w;
    }

    std::vector<data_t> sorted_w(w);
    std::sort(sorted_w.begin(), sorted_w.end());
    data_t w_sum = sorted_w[0];
    data_t l = 1.0 - sorted_w[0];
    for (unsigned int i = 1; i < w.size(); ++i) {
        w_sum += sorted_w[i];
        data_t tmp = 1.0 / (i + 1.0) * (1.0 - w_sum);
        if ((sorted_w[i] + tmp) > 0) {
            l = tmp;
        }
    }

    for (unsigned int i = 0; i < w.size(); ++i) {
        sorted_w[i] = std::max(w[i] + l, 0.0);
    }

    return sorted_w;
}

// inline data_t no_reg(xt::xarray<data_t> &w){
//     return 0.0;
// }

// inline xt::xarray<data_t> no_prox(xt::xarray<data_t> &w, data_t step_size, data_t lambda){
//     return w;
// }

// inline data_t L0_reg(xt::xarray<data_t> &w){
//     data_t cnt = 0;
//     for (unsigned int i = 0; i < w.shape()[0]; ++i) {
//         cnt += (w(i) != 0.0);
//     }

//     return cnt;
// }

// inline xt::xarray<data_t> L0_prox(xt::xarray<data_t> &w, data_t step_size, data_t lambda){
//     data_t tmp = std::sqrt(2 * lambda * step_size);
//     for (unsigned int i = 0; i < w.shape()[0]; ++i) {
//         if (std::abs(w(i)) < tmp)  {
//             w(i) = 0.0;
//         }
//     }
//     return w;
// }

// inline data_t L1_reg(xt::xarray<data_t> &w){
//     data_t cnt = 0;
//     for (unsigned int i = 0; i < w.shape()[0]; ++i) {
//         cnt += std::abs(w(i));
//     }

//     return cnt;
// }

// inline xt::xarray<data_t> L1_prox(xt::xarray<data_t> &w, data_t step_size, data_t lambda){
//     xt::xarray<data_t> sign = xt::sign(w);
//     w = xt::abs(w) - step_size * lambda;
//     return sign*xt::maximum(w,0);
// }

auto reg_from_enum(ENSEMBLE_REGULARIZER reg) {
    if (reg == ENSEMBLE_REGULARIZER::NO) {
        return no_reg;
    } else if (reg == ENSEMBLE_REGULARIZER::L0) {
        return L0_reg;
    } else if (reg == ENSEMBLE_REGULARIZER::L1) {
        return L1_reg;
    } else if (reg == ENSEMBLE_REGULARIZER::hard_L1) {
        return hard_L1_reg;
    } else {
        throw std::runtime_error("Wrong regularizer enum provided. No implementation for this enum found");
    }
}

auto regularizer_from_string(std::string const & regularizer) {
    if (regularizer == "none" || regularizer == "no") {
        return ENSEMBLE_REGULARIZER::NO;
    } else if (regularizer  == "L0") {
        return ENSEMBLE_REGULARIZER::L0;
    } else if (regularizer == "L1") {
        return ENSEMBLE_REGULARIZER::L1;
    } else if (regularizer == "hard_L1" || regularizer == "hard-L1") {
        return ENSEMBLE_REGULARIZER::hard_L1;
    } else {
        throw std::runtime_error("Currently only the three regularizer {none, L0, L1, hard_L1} are supported, but you provided: " + regularizer);
    }
}

#endif