#pragma once

#include <stdexcept>
#include <string>
#include <zlib.h>
#include <string.h>
#include <cstring>
#include "lz4.h"

extern "C" {
#include "shoco.h"
}

namespace DISTANCE {

unsigned int zlib_len(char const * const data, unsigned int n_bytes, int level = Z_BEST_SPEED) {
    // Use zlib's compressBound function to calculate the maximum size of the compressed data buffer
    auto n_bytes_compressed = compressBound(n_bytes);
    char outbuffer[n_bytes_compressed];

    // Perform the compression
    int result = compress2((Bytef*)outbuffer, &n_bytes_compressed, (const Bytef*)data, n_bytes, level);

    if (result != Z_OK) {
        throw(std::runtime_error("Error during zlib compression. Error code was " + std::to_string(result)));
    }

    return n_bytes_compressed;
}

unsigned int shoco_len(char const * const data, unsigned int n_bytes) {
    char outbuffer[n_bytes];
    auto n_bytes_compressed = shoco_compress(data, n_bytes, outbuffer, n_bytes);

    if (n_bytes_compressed > n_bytes) {
        throw(std::runtime_error("Error during shoco compression. Likely the buffer size was not large enough and the compressed string is larger than the original!"));
    }

    return n_bytes_compressed;
}

unsigned int lz4_len(char const * const data, unsigned int n_bytes) {
    char outbuffer[n_bytes];
    int n_bytes_compressed = LZ4_compress_default(data, outbuffer, n_bytes, n_bytes);

    if (n_bytes_compressed < 1) {
        throw(std::runtime_error("Error during lz4 compression. Likely the buffer size was not large enough and the compressed string is larger than the original!"));
    }

    return n_bytes_compressed;
}

template <typename data_t, DDT::DISTANCE distance_type>
internal_t distance(matrix1d<data_t> const &x1, matrix1d<data_t> const &x2) {
    const char * d1 = reinterpret_cast<const char *>(x1.begin());
    unsigned int n1 = sizeof(data_t)*x1.dim;
    unsigned int len_n1;
    if constexpr (distance_type == DISTANCE::TYPES::SHOCO) len_n1 = shoco_len(d1, n1);
    else if constexpr(distance_type == DISTANCE::TYPES::LZ4) len_n1 = lz4_len(d1, n1);
    else len_n1 = zlib_len(d1, n1);

    const char * d2 = reinterpret_cast<const char *>(x2.begin());
    unsigned int n2 = sizeof(data_t)*x2.dim;
    unsigned int len_n2;
    if constexpr (distance_type == DISTANCE::TYPES::SHOCO) len_n2 = shoco_len(d2, n2);
    else if constexpr(distance_type == DISTANCE::TYPES::LZ4) len_n2 = lz4_len(d2, n2);
    else len_n2 = zlib_len(d2, n2);
    
    // TODO This is inefficient, because we repeaditly call new / delete. One way would be to only create
    // a single buffer, but this can become tricky once we compute the distances over multiple threads
    // as we usually do in DistanceDecisionTree::fit
    char * concat_data = new char[n1+n2];

    std::memcpy(concat_data, d1, n1);
    std::memcpy(concat_data + n1, d2, n2);

    unsigned int len_concat;
    if constexpr (distance_type == DISTANCE::TYPES::SHOCO) len_concat = shoco_len(concat_data, n1+n2);
    else if constexpr(distance_type == DISTANCE::TYPES::LZ4) len_concat = lz4_len(concat_data, n1+n2);
    else len_concat = zlib_len(concat_data, n1+n2);

    delete[] concat_data;
    return static_cast<internal_t>(len_concat - std::min(len_n1, len_n2)) / static_cast<internal_t>(std::max(len_n1, len_n2));
}

template <typename data_t>
internal_t distance<DDT::DISTANCE::EUCLIDEAN>(matrix1d<data_t> const &x1, matrix1d<data_t> const &x2) {
    return std::inner_product(x1.begin(), x1.end(), x2.begin(), data_t(0), 
        std::plus<data_t>(), [](data_t x,data_t y){return (y-x)*(y-x);}
    );
}



}