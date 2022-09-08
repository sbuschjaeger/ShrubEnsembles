#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <string>

namespace OPTIMIZER {


enum OPTIMIZER_TYPE {NONE, SGD, ADAM};

enum STEP_SIZE_TYPE {CONSTANT, ADAPTIVE};

auto step_type_from_string(std::string const & step_size_type) {
    if (step_size_type == "CONSTANT" || step_size_type == "constant") {
        return STEP_SIZE_TYPE::CONSTANT;
    } else if (step_size_type == "ADAPTIVE" || step_size_type == "adaptive") {
        return STEP_SIZE_TYPE::ADAPTIVE;
    } else {
        throw std::runtime_error("Currently only step_size_type {CONSTANT, ADAPTIVE} are supported, but you provided: " + step_size_type);
    }
}

auto optimizer_from_string(std::string const & optimizer) {
    if (optimizer == "SGD" || optimizer == "sgd") {
        return OPTIMIZER_TYPE::SGD;
    } else if (optimizer == "ADAM" || optimizer == "adam") {
        return OPTIMIZER_TYPE::ADAM;
    } else if (optimizer == "NONE" || optimizer == "none") {
        return OPTIMIZER_TYPE::NONE;
    } else {
        throw std::runtime_error("Currently only optimizer {ADAM, SGD, NONE} are supported, but you provided: " + optimizer);
    }
}

template<OPTIMIZER_TYPE, STEP_SIZE_TYPE>
struct Optimizer{
    /* This should never be reached */
};

template<STEP_SIZE_TYPE step_size_type> 
struct Optimizer<OPTIMIZER_TYPE::NONE,step_size_type> {
    Optimizer(internal_t step_size) {}

    void reset() {}

    void step(std::vector<internal_t> &weight, std::vector<internal_t> const &grad) {}

    unsigned int num_bytes() const {
        return sizeof(*this);
    }
};

template<STEP_SIZE_TYPE step_size_type> 
struct Optimizer<OPTIMIZER_TYPE::SGD,step_size_type> {
    internal_t step_size;

    Optimizer(internal_t step_size) : step_size(step_size){}
    Optimizer() : step_size(1e-1) {}

    void reset() {}

    void step(std::vector<internal_t> &weight, std::vector<internal_t> const &grad) {
        internal_t step_size;
        if constexpr(step_size_type == STEP_SIZE_TYPE::ADAPTIVE) {
            step_size = 1.0 / (weight.size() + 1);
        } else {
            step_size = this->step_size;
        }

        for (unsigned int i = 0; i < grad.size(); ++i) {
            weight[i] -= step_size * grad[i]; 
        }
    }

    unsigned int num_bytes() const {
        return sizeof(*this);
    }
};

template<STEP_SIZE_TYPE step_size_type> 
struct Optimizer<OPTIMIZER_TYPE::ADAM,step_size_type> {
    internal_t step_size;
    internal_t beta1;
    internal_t beta2;

    std::vector<internal_t> m;
    std::vector<internal_t> v;
    unsigned int t;

    Optimizer(internal_t step_size, internal_t beta1, internal_t beta2) : step_size(step_size), beta1(beta1), beta2(beta2), t(1) {}

    Optimizer(internal_t step_size) : step_size(step_size), beta1(0.9), beta2(0.999), t(1) {}

    Optimizer() : step_size(1e-1), beta1(0.9), beta2(0.999), t(1) {}

    void reset() {
        m.clear();
        v.clear();
        t = 1;
    }

    unsigned int num_bytes() const {
        return sizeof(*this) + m.size() * sizeof(internal_t) + v.size()*sizeof(internal_t);
    }

    void step(std::vector<internal_t> &weight, std::vector<internal_t> const &grad) {
        data_t step_size;

        if constexpr(step_size_type == STEP_SIZE_TYPE::ADAPTIVE) {
            step_size = 1.0 / (weight.size() + 1);
        } else {
            step_size = this->step_size;
        }

        // TODO Also check v / weight?
        if (m.size() != grad.size()) {
            m = std::vector<internal_t>(grad.size(), 0);
            v = std::vector<internal_t>(grad.size(), 0);
            t = 1;
        }

        for (unsigned int i = 0; i < grad.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta1 * v[i] + (1.0 - beta1) * grad[i] * grad[i];
            internal_t m_cor = m[i] / (1.0 - std::pow(beta1,t));
            internal_t v_cor = v[i] / (1.0 - std::pow(beta2, t));
            weight[i] += -step_size * m_cor / (std::sqrt(v_cor) + 1e-7);
        }
        t += 1; 
    }
};

}

#endif