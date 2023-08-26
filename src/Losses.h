#pragma once

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "Datatypes.h"
#include "Matrix.h"

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

class Loss {
public:
    virtual void operator()(matrix1d<internal_t> const & pred, matrix1d<internal_t> & losses, unsigned int target) = 0;
    
    virtual void deriv(matrix1d<internal_t> const & pred, matrix1d<internal_t> & grad, unsigned int target) = 0;

    virtual std::unique_ptr<Loss> clone() const = 0;

    virtual ~Loss() {}

    matrix2d<internal_t> operator()(matrix2d<internal_t> const &pred, matrix1d<unsigned int> const &target) {
        matrix2d<internal_t> losses(pred.rows, pred.cols);
        std::fill(losses.begin(), losses.end(), 0);

        for (unsigned int i = 0; i < pred.rows; ++i) {
            auto iloss = losses(i);
            (*this)(pred(i), iloss, target(i));
        }

        return losses;
    };

    matrix2d<internal_t> deriv(matrix2d<internal_t> const & pred, matrix1d<internal_t> & grad, matrix1d<unsigned int> &target) {
        matrix2d<internal_t> loss_deriv(pred.rows, pred.cols);
        std::fill(loss_deriv.begin(), loss_deriv.end(), 0);

        for (unsigned int i = 0; i < pred.rows; ++i) {
            auto ideriv = loss_deriv(i);
            this->deriv(pred(i), ideriv, target(i));
        }

        return loss_deriv;
    };
};

class MSE : public Loss {
public:
    using Loss::operator();
    using Loss::deriv;

    void operator()(matrix1d<internal_t> const & pred, matrix1d<internal_t> & losses, unsigned int target) {
        for (unsigned int i = 0; i < pred.dim; ++i) {
            if (i == target) {
                losses(i) = (pred(i) - 1.0) * (pred(i) - 1.0);
            } else {
                losses(i) = (pred(i) - 0.0) * (pred(i) - 0.0);
            }
        }
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

    std::unique_ptr<Loss> clone() const {
        return std::make_unique<MSE>();
    }

    ~MSE() {}
};

class CrossEntropy : public Loss {
public:
    using Loss::operator();
    using Loss::deriv;

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

    void operator()(matrix1d<internal_t> const & pred, matrix1d<internal_t> & losses, unsigned int target) {
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

    std::unique_ptr<Loss> clone() const {
        return std::make_unique<CrossEntropy>();
    }

    ~CrossEntropy() {}
};