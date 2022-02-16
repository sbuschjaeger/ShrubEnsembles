#ifndef LOSSES_H
#define LOSSES_H

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "Datatypes.h"

namespace LOSS {

enum class TYPE {CROSS_ENTROPY, MSE};

std::vector<internal_t> softmax(std::vector<internal_t> const &x) {
    std::vector<internal_t> tmp(x);
    internal_t m = *std::max_element(tmp.begin(), tmp.end());

    for(unsigned int i = 0; i < tmp.size(); i++) {
        tmp[i] = std::exp(tmp[i] - m);
    } 

    internal_t sum = static_cast<internal_t>(std::accumulate(tmp.begin(), tmp.end(), 0.0));
    std::transform(tmp.begin(), tmp.end(), tmp.begin(), [sum](internal_t xi){ return xi/sum; });

    return tmp;
}


/**
 * @brief  The softmax function which maps the input tensor to probabilities. The shape is assumed to be (batch_size, n_classes). Softmax is applied for each row of the input matrix.
 * @note   
 * @param  &X: Inputmatrix over which softmax will be applied. Assumed to have shape (batch_size, n_classes) 
 * @retval A new matrix with shape (batch_size, n_classes) where softmax has been applied to every row.
 */
std::vector<std::vector<internal_t>> softmax(std::vector<std::vector<internal_t>> const &pred) {
    std::vector<std::vector<internal_t>> p(pred.size());

    for (unsigned int i = 0; i < pred.size(); ++i) {
        p[i] = softmax(pred[i]);
    }

    return p;
}

template<LOSS::TYPE>
struct Loss{
    /* This should never be reached */
};

template<>
struct Loss<LOSS::TYPE::MSE> {

    void loss(internal_t const * const pred, internal_t * const losses, unsigned int target, unsigned int n_classes) {
        for (unsigned int i = 0; i < n_classes; ++i) {
            if (i == target) {
                losses[i] = (pred[i] - 1.0) * (pred[i] - 1.0);
            } else {
                losses[i] = (pred[i] - 0.0) * (pred[i] - 0.0);
            }
        }
    }

    /**
     * @brief  Computes the mse loss. 
     * @note   This is not the regular MSE loss, but it maps the prediction to probabilities beforehand using softmax!
     * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
     * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
     * @retval The mse loss for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
     */
    std::vector<std::vector<internal_t>> loss(std::vector<std::vector<internal_t>> const &pred, std::vector<unsigned int> const &target) {
        std::vector<std::vector<internal_t>> losses(
            pred.size(),
            std::vector<internal_t>(pred[0].size(), 0)
        );

        for (unsigned int i = 0; i < pred.size(); ++i) {
            loss(&pred[i][0], &losses[i][0], target[i], pred[i].size());
        }

        return losses;
    }

    void deriv(internal_t const * const pred, internal_t * const grad, unsigned int target, unsigned int n_classes) {
        for (unsigned int i = 0; i < n_classes; ++i) {
            if (i == target) {
                grad[i] = 2 * (pred[i] - 1.0);
            } else {
                grad[i] = 2 * (pred[i] - 0.0);
            }
        }
    }

    std::vector<std::vector<internal_t>> deriv(std::vector<std::vector<internal_t>> const &pred, std::vector<unsigned int> const &target) {
        std::vector<std::vector<internal_t>> loss_deriv(
            pred.size(),
            std::vector<internal_t>(pred[0].size(), 0)
        );

        for (unsigned int i = 0; i < pred.size(); ++i) {
            deriv(&pred[i][0], &loss_deriv[i][0], target[i], pred[i].size());
        }

        return loss_deriv;
    }
    
};

template<>
struct Loss<LOSS::TYPE::CROSS_ENTROPY> {

    void softmax(internal_t * const x, unsigned int dim) {
        internal_t m = *std::max_element(&x[0], &x[dim]);

        for(unsigned int i = 0; i < dim; i++) {
            x[i] = std::exp(x[i] - m);
        } 

        internal_t sum = static_cast<internal_t>(std::accumulate(&x[0], &x[dim], 0.0));
        std::transform(&x[0], &x[dim], &x[0], [sum](internal_t xi){ return xi/sum; });
    }

    void deriv(internal_t const * const pred, internal_t * const grad, unsigned int target, unsigned int n_classes) {
        std::copy(&pred[0], &pred[n_classes], &grad[0]);
        softmax(grad, n_classes);

        grad[target] -= 1.0;
    }

    /**
     * @brief  Computes the cross entropy loss, which is the negative log-liklihood combined with the softmax function. The prediction tensor is assumed to have a shape of (batch_size, n_classes), whereas the target vector is assumed to be a vector of shape (batch_size) in which each entry represents the corresponding class.
     * @note   
     * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
     * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
     * @retval The cross-entropy for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
     */
    std::vector<std::vector<internal_t>> deriv(std::vector<std::vector<internal_t>> const &pred, std::vector<unsigned int> const &target) {
        std::vector<std::vector<internal_t>> loss_deriv(
            pred.size(),
            std::vector<internal_t>(pred[0].size(), 0)
        );

        for (unsigned int i = 0; i < pred.size(); ++i) {
            deriv(&pred[i][0], &loss_deriv[i][0], target[i], pred[i].size());
        }

        return loss_deriv;
        
        // std::vector<std::vector<internal_t>> p = softmax(pred);

        // for (unsigned int i = 0; i < p.size(); ++i) {
        //     p[i][target[i]] -= 1.0;
        // }

        // return p;
    }

    void loss(internal_t const * const pred, internal_t * const losses, unsigned int target, unsigned int n_classes) {
        std::copy(&pred[0], &pred[n_classes], &losses[0]);
        softmax(losses, n_classes);

        for (unsigned int j = 0; j < n_classes; ++j) {
            if (j == target) {
                losses[j] = -1.0 * std::log(losses[j]);
            } else {
                losses[j] = 0;
            }
        }
    }

    /**
     * @brief  The first derivation of the cross entropy loss. The prediction tensor is assumed to have a shape of (batch_size, n_classes), whereas the target vector is assumed to be a vector of shape (batch_size) in which each entry represents the corresponding class.
     * @note   
     * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
     * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
     * @retval The first derivation of the cross-entropy for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
     */
    std::vector<std::vector<internal_t>> loss(std::vector<std::vector<internal_t>> const &pred, std::vector<unsigned int> const &target) {
        std::vector<std::vector<internal_t>> losses(
            pred.size(),
            std::vector<internal_t>(pred[0].size(), 0)
        );

        for (unsigned int i = 0; i < pred.size(); ++i) {
            deriv(&pred[i][0], &losses[i][0], target[i], pred[i].size());
        }

        return losses;

        // std::vector<std::vector<internal_t>> p = softmax(pred);

        // for (unsigned int i = 0; i < pred.size(); ++i) {
        //     for (unsigned int j = 0; j < pred[i].size(); ++j) {
        //         if (j == target[i]) {
        //             p[i][j] = -1.0 * std::log(p[i][j]);
        //         } else {
        //             p[i][j] = 0;
        //         }
        //     }
        // }

        // return p;
    }
};




/**
 * @brief  Computes the first derivative of the mse loss. Contrary to common implementations of the mse, this version first scales the input to probabilities using the softmax!
 * @note   This is not the regular MSE loss, but it maps the prediction to probabilities beforehand using softmax!
 * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
 * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
 * @retval The first derivative of the mse loss for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
 */
// auto from_enum(TYPE loss) {
//     if (loss == TYPE::CROSS_ENTROPY) {
//        return Loss<TYPE::CROSS_ENTROPY>();
//     } else if (loss == TYPE::MSE) {
//         return Loss<TYPE::MSE>();
//     } else {
//         throw std::runtime_error("Wrong loss enum provided. No implementation for this enum found");
//     }
// }

// auto deriv_from_enum(TYPE loss) {
//     if (loss == TYPE::CROSS_ENTROPY) {
//         return cross_entropy_deriv;
//     } else if (loss == TYPE::MSE) {
//         return mse_deriv;
//     } else {
//         throw std::runtime_error("Wrong loss enum provided. No implementation for this enum found");
//     }
// }

auto from_string(std::string const & loss) {
    if (loss == "cross-entropy") {
        return TYPE::CROSS_ENTROPY;
    } else if (loss  == "mse") {
        return TYPE::MSE;
    } else {
        throw std::runtime_error("Currently only the two losses {cross-entropy, mse} are supported, but you provided: " + loss);
    }
}

}

#endif