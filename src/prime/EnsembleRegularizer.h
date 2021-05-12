#ifndef ENSEMBLE_REGULARIZER_H
#define ENSEMBLE_REGULARIZER_H

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "Datatypes.h"


namespace ENSEMBLE_REGULARIZER {

enum class TYPE {NO,L0,L1,hard_L0};

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

std::vector<data_t> hard_L0_reg(std::vector<data_t> const &w, data_t K) {
    std::vector<unsigned int> top_idx = top_k(w, K);
    std::vector<data_t> tmp_w(w.size(), 0);

    for (auto i : top_idx) {
        tmp_w[i] = w[i];
    }

    return tmp_w;
}

std::vector<data_t> to_prob_simplex(std::vector<data_t> const &w) {
    if (w.size() == 0) {
        return w;
    }

    std::vector<data_t> u(w);
    std::sort(u.begin(), u.end(), std::greater<int>());

    data_t u_sum = 0; 
    data_t l = 0;
    for (unsigned int i = 0; i < w.size(); ++i) {
        u_sum += u[i];
        data_t tmp = 1.0 / (i + 1.0) * (1.0 - u_sum);
        if ((u[i] + tmp) > 0) {
            l = tmp;
        }
    }

    for (unsigned int i = 0; i < w.size(); ++i) {
        u[i] = std::max(w[i] + l, 0.0);
    }

    return u;
}

auto from_enum(TYPE reg) {
    if (reg == TYPE::NO) {
        return no_reg;
    } else if (reg == TYPE::L0) {
        return L0_reg;
    } else if (reg == TYPE::L1) {
        return L1_reg;
    } else if (reg == TYPE::hard_L0) {
        return hard_L0_reg;
    } else {
        throw std::runtime_error("Wrong regularizer enum provided. No implementation for this enum found");
    }
}

auto from_string(std::string const & regularizer) {
    if (regularizer == "none" || regularizer == "no") {
        return TYPE::NO;
    } else if (regularizer  == "L0") {
        return TYPE::L0;
    } else if (regularizer == "L1") {
        return TYPE::L1;
    } else if (regularizer == "hard_L0" || regularizer == "hard-L0") {
        return TYPE::hard_L0;
    } else {
        throw std::runtime_error("Currently only the three regularizer {none, L0, L1, hard_L0} are supported, but you provided: " + regularizer);
    }
}
}

#endif