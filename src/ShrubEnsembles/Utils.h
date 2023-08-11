#pragma once

/**
 * @brief  Scales the given matrix X in-place by the given factor s
 * @note   
 * @param  &X: The matrix
 * @param  s: The scaling factor
 * @retval None, the operation changes X in-place
 */
template <typename data_t>
void scale(matrix2d<data_t> &X, data_t s) {
    for (unsigned int j = 0; j < X.rows; ++j) {
        for (unsigned int k = 0; k < X.cols; ++k) {
            X(j,k) *= s;
        }
    }
}

/**
 * @brief  Computes of all values in the given matrix X.
 * @note   
 * @param  &X: The matrix
 * @retval The mean of all matrix entries. 
 */
template <typename data_t>
data_t mean_all_dim(matrix2d<data_t> &X) {
    unsigned int n_first = X.rows;
    unsigned int n_second = X.cols;
    data_t mean = 0;

    for (unsigned int j = 0; j < n_first; ++j) {
        for (unsigned int k = 0; k < n_second; ++k) {
            mean += X(j,k);
        }
    }

    return mean / (n_first * n_second);
}

/**
 * @brief Computes thes weighted sum across the first dimension of the given tensor using the supplied weights
 * @note   
 * @param  &X: A (N,M,K) tensor
 * @param  &weights: A (N,) vector
 * @retval A (M,K) matrix stored as matrix<data_t>
 */
template <typename data_t>
matrix2d<data_t> weighted_sum_first_dim(matrix3d<data_t> const &X, matrix1d<data_t> const &weights) {
    matrix2d<data_t> XMean(X.ny, X.nz); 
    std::fill(XMean.begin(), XMean.end(), 0);

    for (unsigned int i = 0; i < X.nx; ++i) {
        for (unsigned int j = 0; j < X.ny; ++j) {
            for (unsigned int k = 0; k < X.nz; ++k) {
                // XMean[j][k] += X[i][j][k] * weights[i];
                XMean(j,k) += X(i,j,k) * weights(i); //[i];
            }
        }
    }

    return XMean;
}

matrix1d<unsigned int> sample_indices(unsigned int n_data, unsigned int batch_size, bool bootstrap, std::minstd_rand &gen) {
    if (batch_size >= n_data || batch_size == 0) {
        batch_size = n_data;
    }

    matrix1d<unsigned int> idx(batch_size);
    if (bootstrap) {
        std::uniform_int_distribution<> dist(0, n_data - 1); 
        for (unsigned int i = 0; i < batch_size; ++i) {
            idx(i) = dist(gen);
        }
    } else {
        matrix1d<unsigned int> _idx(n_data);
        std::iota(_idx.begin(), _idx.end(), 0);
        std::shuffle(_idx.begin(), _idx.end(), gen);

        for (unsigned int i = 0; i < batch_size; ++i) {
            idx(i) = _idx(i);
        }
    }
    return idx;
}

matrix1d<unsigned int> sample_indices(matrix1d<unsigned int> const &idx, unsigned int batch_size, bool bootstrap, std::minstd_rand &gen) {
    unsigned int n_data = idx.dim;
    if (batch_size >= n_data || batch_size == 0) {
        batch_size = n_data;
    }
    matrix1d<unsigned int> new_idx(batch_size);
    if (bootstrap) {
        std::uniform_int_distribution<> dist(0, n_data - 1); 
        for (unsigned int i = 0; i < batch_size; ++i) {
            new_idx(i) = idx(dist(gen));
        }
    } else {
        matrix1d<internal_t> _idx(n_data);
        std::iota(_idx.begin(), _idx.end(), 0);
        std::shuffle(_idx.begin(), _idx.end(), gen);

        for (unsigned int i = 0; i < batch_size; ++i) {
            new_idx(i) = idx(_idx(i));
        }
    }
    return new_idx;
}

matrix1d<unsigned int> sample_indices(unsigned int n_data, unsigned int batch_size, bool bootstrap, unsigned long seed = 12345L) {
    std::minstd_rand gen(seed);
    return sample_indices(n_data,batch_size,bootstrap,gen);
}

matrix1d<unsigned int> sample_indices(matrix1d<unsigned int> const &idx, unsigned int batch_size, bool bootstrap, unsigned long seed = 12345L) {
    std::minstd_rand gen(seed);
    return sample_indices(idx,batch_size,bootstrap,gen);
}