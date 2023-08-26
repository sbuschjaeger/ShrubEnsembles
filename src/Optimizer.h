#pragma once

#include <vector>
#include <cmath>
#include <unordered_map>

#include "Matrix.h"
#include "Datatypes.h"

class Optimizer {
public:
    virtual void reset() = 0;

    // id can be used to identify groups of parameters
    virtual void step(unsigned int id, matrix1d<internal_t> &weight, matrix1d<internal_t> const &grad) = 0;

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

    void step(unsigned int id, matrix1d<internal_t> &weight, matrix1d<internal_t> const &grad) {
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
    
    // TODO replace std::vector with matrix1d
    std::unordered_map<unsigned int, std::vector<internal_t>> ms;
    std::unordered_map<unsigned int, std::vector<internal_t>> vs;

    // TODO Use this?
    // matrix1d<internal_t> m;
    // matrix1d<internal_t> v;
    unsigned int t;

    Adam(internal_t step_size, internal_t beta1, internal_t beta2) : step_size(step_size), beta1(beta1), beta2(beta2), t(1) {}

    Adam(internal_t step_size) : step_size(step_size), beta1(0.9), beta2(0.999), t(1) {}

    Adam() : step_size(1e-1), beta1(0.9), beta2(0.999), t(1) {}

    void reset() {
        ms.clear();
        vs.clear();
        t = 1;
    }

    unsigned int num_bytes() const {
        auto ms_size = ms.size() > 0 ? ms.size() * ms.begin()->second.size() * sizeof(internal_t) : 0;
        auto vs_size = vs.size() > 0 ? vs.size() * vs.begin()->second.size() * sizeof(internal_t) : 0;
        return sizeof(*this) +  ms_size + vs_size;
    }

    void step(unsigned int id, matrix1d<internal_t> &weight, matrix1d<internal_t> const &grad) {
        
        auto me = ms.find(id);
        if (me == ms.end() || me->second.size() != grad.dim) {
            ms[id] = std::vector<internal_t>(grad.dim, 0);
            t = 1;
        } 

        auto ve = vs.find(id);
        if (ve == vs.end() || ve->second.size() != grad.dim) {
            vs[id] = std::vector<internal_t>(grad.dim, 0);
            t = 1;
        } 
        
        auto & m = ms[id];
        auto & v = vs[id];

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
        a->ms = this->ms;
        a->vs = this->vs;

        return a;
    }
};
