#pragma once

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "Datatypes.h"
#include "Matrix.h"

namespace LOSS {

enum class TYPE {CROSS_ENTROPY, MSE};

// std::vector<internal_t> softmax(std::vector<internal_t> const &x) {
// matrix1d<internal_t> softmax(matrix1d<internal_t> const &x) {
//     matrix1d<internal_t> tmp(x.dim);
//     std::copy(x.begin(), x.end(), tmp.begin());

//     internal_t m = *std::max_element(tmp.begin(), tmp.end());

//     for(unsigned int i = 0; i < tmp.dim; i++) {
//         tmp(i) = std::exp(tmp(i) - m);
//     } 

//     internal_t sum = static_cast<internal_t>(std::accumulate(tmp.begin(), tmp.end(), 0.0));
//     std::transform(tmp.begin(), tmp.end(), tmp.begin(), [sum](internal_t xi){ return xi/sum; });

//     return tmp;
// }

// void softmax(matrix1d<internal_t> &out, matrix1d<internal_t> const &x) {
//     std::copy(x.begin(), x.end(), out.begin());

//     internal_t m = *std::max_element(out.begin(), out.end());

//     for(unsigned int i = 0; i < out.dim; i++) {
//         out(i) = std::exp(out(i) - m);
//     } 

//     internal_t sum = static_cast<internal_t>(std::accumulate(out.begin(), out.end(), 0.0));
//     std::transform(out.begin(), out.end(), out.begin(), [sum](internal_t xi){ return xi/sum; });
// }


// /**
//  * @brief  The softmax function which maps the input tensor to probabilities. The shape is assumed to be (batch_size, n_classes). Softmax is applied for each row of the input matrix.
//  * @note   
//  * @param  &X: Inputmatrix over which softmax will be applied. Assumed to have shape (batch_size, n_classes) 
//  * @retval A new matrix with shape (batch_size, n_classes) where softmax has been applied to every row.
//  */
// matrix2d<internal_t> softmax(matrix2d<internal_t> const &pred) {
//     matrix2d<internal_t> p(pred.rows, pred.cols);

//     for (unsigned int i = 0; i < pred.rows; ++i) {
//         softmax(p(i), pred(i));
//     }

//     return p;
// }

template<LOSS::TYPE>
struct Loss{
    /* This should never be reached */
};

template<>
struct Loss<LOSS::TYPE::MSE> {

    void loss(matrix1d<internal_t> const & pred, matrix1d<internal_t> & losses, unsigned int target) {
        for (unsigned int i = 0; i < pred.dim; ++i) {
            if (i == target) {
                losses(i) = (pred(i) - 1.0) * (pred(i) - 1.0);
            } else {
                losses(i) = (pred(i) - 0.0) * (pred(i) - 0.0);
            }
        }
    }

    /**
     * @brief  Computes the mse loss. 
     * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
     * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
     * @retval The mse loss for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
     */
    matrix2d<internal_t> loss(matrix2d<internal_t> const &pred, matrix1d<unsigned int> &target) {
        matrix2d<internal_t> losses(pred.rows, pred.cols);
        std::fill(losses.begin(), losses.end(), 0);

        for (unsigned int i = 0; i < pred.rows; ++i) {
            auto iloss = losses(i);
            loss(pred(i), iloss, target(i));
        }

        return losses;
    }

    void deriv(matrix1d<internal_t> const & pred, matrix1d<internal_t> & grad, unsigned int target) {
        for (unsigned int i = 0; i < pred.dim; ++i) {
            if (i == target) {
                grad(i) = 2 * (pred(i) - 1.0);
            } else {
                grad(i) = 2 * (pred(i) - 0.0);
            }
        }
    }

    matrix2d<internal_t> deriv(matrix2d<internal_t> const &pred, matrix1d<unsigned int> &target) {
        matrix2d<internal_t> loss_deriv(pred.rows, pred.cols);
        std::fill(loss_deriv.begin(), loss_deriv.end(), 0);

        for (unsigned int i = 0; i < pred.rows; ++i) {
            auto ideriv = loss_deriv(i);

            deriv(pred(i), ideriv, target(i));
        }

        return loss_deriv;
    }
    
};

template<>
struct Loss<LOSS::TYPE::CROSS_ENTROPY> {

    // void softmax(internal_t * const x {
    // internal_t m = *std::max_element(&x[0], &x[dim]);
    void softmax(matrix1d<internal_t> & x) {
        internal_t m = *std::max_element(x.begin(), x.end());

        for(unsigned int i = 0; i < x.dim; i++) {
            x(i) = std::exp(x(i) - m);
        } 

        internal_t sum = static_cast<internal_t>(std::accumulate(x.begin(), x.end(), 0.0));
        std::transform(x.begin(), x.end(), x.begin(), [sum](internal_t xi){ return xi/sum; });
    }

    void deriv(matrix1d<internal_t> const & pred, matrix1d<internal_t> & grad, unsigned int target) {
        std::copy(pred.begin(), pred.end(), grad.begin());
        softmax(grad);

        grad(target) -= 1.0;
    }

    /**
     * @brief  Computes the cross entropy loss, which is the negative log-liklihood combined with the softmax function. The prediction tensor is assumed to have a shape of (batch_size, n_classes), whereas the target vector is assumed to be a vector of shape (batch_size) in which each entry represents the corresponding class.
     * @note   
     * @param  &pred: The per-class prediction tensor for each sample. This tensor is assumed to have a shape of (batch_size, n_classes)
     * @param  &target: The per-sample target which is assumed to be from {0,\dots,n_classes - 1}. This tensor is assumed to have a shape of (batch_size)
     * @retval The cross-entropy for each class and each sample. The return tensor has a shape of (batch_size, n_classes). 
     */
    //std::vector<std::vector<internal_t>> deriv(std::vector<std::vector<internal_t>> const &pred, std::vector<unsigned int> const &target) {
    matrix2d<internal_t> deriv(matrix2d<internal_t> const &pred, matrix1d<unsigned int> &target) {
        matrix2d<internal_t> loss_deriv(pred.rows, pred.cols);
        std::fill(loss_deriv.begin(), loss_deriv.end(), 0);

        for (unsigned int i = 0; i < pred.rows; ++i) {
            auto ideriv = loss_deriv(i);
            deriv(pred(i), ideriv, target(i));
        }

        return loss_deriv;
    }

    void loss(matrix1d<internal_t> const & pred, matrix1d<internal_t> & losses, unsigned int target) {
        std::copy(pred.begin(), pred.end(), losses.begin());
        softmax(losses);

        for (unsigned int j = 0; j < pred.dim; ++j) {
            if (j == target) {
                losses(j) = -1.0 * std::log(losses(j));
            } else {
                losses(j) = 0;
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
    matrix2d<internal_t> loss(matrix2d<internal_t> const &pred, matrix1d<unsigned int> &target) {
        matrix2d<internal_t> losses(pred.rows, pred.cols);
        std::fill(losses.begin(), losses.end(), 0);
        
        for (unsigned int i = 0; i < pred.rows; ++i) {
            auto iloss = losses(i);
            deriv(pred(i), iloss, target(i));
        }

        return losses;
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