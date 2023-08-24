#pragma once

#include <vector>
#include <cmath>

#include "Matrix.h"
#include "Datatypes.h"

class Optimizer {
public:
    virtual void reset() = 0;

    virtual void step(matrix1d<internal_t> &weight, matrix1d<internal_t> const &grad) = 0;

    virtual unsigned int num_bytes() const = 0;

    virtual ~Optimizer() { }

    virtual std::unique_ptr<Optimizer> clone() const = 0;
};

class SGD : public Optimizer {
public:
    internal_t step_size;

    SGD(internal_t step_size) : step_size(step_size){}

    SGD() : step_size(1e-1) {}

    void reset() {}

    void step(matrix1d<internal_t> &weight, matrix1d<internal_t> const &grad) {
        for (unsigned int i = 0; i < grad.dim; ++i) {
            weight(i) -= step_size * grad(i); 
        }
    }

    unsigned int num_bytes() const {
        return sizeof(*this);
    }

    ~SGD() {}

    std::unique_ptr<Optimizer> clone() const {
        return std::make_unique<SGD>(step_size);
    }
};

class Adam : public Optimizer {
public:
    internal_t step_size;
    internal_t beta1;
    internal_t beta2;

    std::vector<internal_t> m;
    std::vector<internal_t> v;

    // TODO Use this?
    // matrix1d<internal_t> m;
    // matrix1d<internal_t> v;
    unsigned int t;

    Adam(internal_t step_size, internal_t beta1, internal_t beta2) : step_size(step_size), beta1(beta1), beta2(beta2), t(1) {}

    Adam(internal_t step_size) : step_size(step_size), beta1(0.9), beta2(0.999), t(1) {}

    Adam() : step_size(1e-1), beta1(0.9), beta2(0.999), t(1) {}

    void reset() {
        m.clear();
        v.clear();
        t = 1;
    }

    unsigned int num_bytes() const {
        return sizeof(*this) + m.size() * sizeof(internal_t) + v.size()*sizeof(internal_t);
    }

    void step(matrix1d<internal_t> &weight, matrix1d<internal_t> const &grad) {
        // TODO Also check v / weight?
        if (m.size() != grad.dim) {
            // m = matrix1d<internal_t>(grad.dim);
            // std::fill(m.begin(), m.end(), 0);

            // v = matrix1d<internal_t>(grad.dim);
            // std::fill(v.begin(), v.end(), 0);

            m = std::vector<internal_t>(grad.dim, 0);
            v = std::vector<internal_t>(grad.dim, 0);
            t = 1;
        }

        for (unsigned int i = 0; i < grad.dim; ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad(i);
            v[i] = beta1 * v[i] + (1.0 - beta1) * grad(i) * grad(i);
            internal_t m_cor = m[i] / (1.0 - std::pow(beta1,t));
            internal_t v_cor = v[i] / (1.0 - std::pow(beta2, t));
            weight(i) += -step_size * m_cor / (std::sqrt(v_cor) + 1e-7);
        }
        t += 1; 
    }

    ~Adam() {}

    std::unique_ptr<Optimizer> clone() const {
        std::unique_ptr<Adam> a = std::make_unique<Adam>(step_size, beta1, beta2);
        a->t = this->t;
        
        a->m = this->m;
        a->v = this->v;

        return a;
    }
};
