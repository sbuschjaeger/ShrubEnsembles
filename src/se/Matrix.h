#pragma once

#include <memory>

template<typename T>
class matrix1d {

public:
    unsigned int dim;
    std::unique_ptr<T[]> data;
    bool has_ownership;

    // Default c'tor for pybind
    matrix1d() : dim(0), data(nullptr), has_ownership(false) {}

    matrix1d(unsigned int dim, T * const src, bool take_ownership = true) 
    : dim(dim), data(src), has_ownership(take_ownership) {}

    matrix1d(unsigned int dim) 
    : dim(dim), data(new T[dim]), has_ownership(true) {}
    
    matrix1d( matrix1d&& a ) : dim(a.dim), data( std::move( a.data ) ) {}

    matrix1d( matrix1d& a ) = delete;
    matrix1d& operator=(const matrix1d& other) = delete;
    matrix1d& operator=(const matrix1d&& other) = delete;

    ~matrix1d() {
        if (!has_ownership) {
            data.release();
        }
    }

    auto begin() const {
        return &data[0];
    }

    auto begin() {
        return &data.get()[0];
    }

    auto end() const {
        return &data[dim];
    }

    auto end() {
        return &data.get()[dim];
    }

    T& operator()(unsigned int i) {
        return data[i];
    }

    T operator()(unsigned int i) const {
        return data.get()[i];
    }
};

template<typename T>
class matrix2d {
public:
    unsigned int rows;
    unsigned int cols;
    std::unique_ptr<T[]> data;
    bool has_ownership;

    // Default c'tor for pybind
    matrix2d() : rows(0), cols(0), data(nullptr), has_ownership(false) {}

    matrix2d(unsigned int rows, unsigned int columns, T * const src, bool take_ownership = true) 
    : rows(rows), cols(columns), data(src), has_ownership(take_ownership) {
    }

    matrix2d(unsigned int rows, unsigned int columns) 
    : rows(rows), cols(columns), data(new T[rows * columns]), has_ownership(true) { }

    //matrix2d( matrix2d&& a ) = delete;
    matrix2d( matrix2d&& a ) : rows(a.rows), cols(a.cols), data( std::move( a.data ) ), has_ownership(a.has_ownership){}

    matrix2d( matrix2d& a ) = delete;
    matrix2d& operator=(const matrix2d& other) = delete;
    matrix2d& operator=(const matrix2d&& other) = delete;

    ~matrix2d() {
        if (!has_ownership) {
            data.release();
        }
    }

    auto begin() const {
        return &data[0];
    }

    auto begin() {
        return &data.get()[0];
    }

    auto end() const {
        return &data[cols * rows];
    }

    auto end() {
        return &data.get()[cols * rows];
    }

    matrix1d<T> operator()(unsigned int i) const {
        return matrix1d<T>(cols,&data.get()[i * cols], false);
    }

    T& operator()(unsigned int i, unsigned int j) {
        return data.get()[i * cols + j];
    }

    T operator()(unsigned int i, unsigned int j) const {
        return data.get()[i * cols + j];
    }
};

template<typename T>
class matrix3d {

public:
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;

    std::unique_ptr<T[]> data;
    bool has_ownership;

    // Default c'tor for pybind
    matrix3d() : nx(0), ny(0), nz(0), data(nullptr), has_ownership(false) {}

    matrix3d(unsigned int nx, unsigned int ny, unsigned int nz, T * const src, bool take_ownership = true) 
    : nx(nx), ny(ny), nz(nz), data(src), has_ownership(take_ownership) {}

    matrix3d(unsigned int nx, unsigned int ny, unsigned int nz) 
    : nx(nx), ny(ny), nz(nz), data(new T[nx * ny * nz]), has_ownership(true) {}

    matrix3d( matrix3d&& a ) : nx(a.nx), ny(a.ny), nz(a.nz), data( std::move( a.data ) ) {}

    matrix3d( matrix3d& a ) = delete;
    matrix3d& operator=(const matrix3d& other) = delete;
    matrix3d& operator=(const matrix3d&& other) = delete;

    ~matrix3d() {
        if (!has_ownership) {
            data.release();
        }
    }

    auto begin() const {
        return &data[0];
    }

    auto begin() {
        return &data[0];
    }

    auto end() const {
        return &data[nx * ny * nz];
    }

    auto end() {
        return &data[nx * ny * nz];
    }


    matrix2d<T> operator()(unsigned int i) {
        matrix2d<T> tmp(ny,nz,&data.get()[i * (ny * nz)], false);

        return tmp;
    }

    matrix1d<T> operator()(unsigned int i, unsigned j) {
        matrix1d<T> tmp(ny,nz,&data[i * (ny * nz) + j * nz], false);

        return tmp;
    }

    T& operator()(unsigned int i, unsigned int j, unsigned int k) {
        return &data[k + j*nz + i*ny*nz];
    }

    T operator()(unsigned int i, unsigned int j, unsigned int k) const {
        return data.get()[k + j*nz + i*ny*nz];
    }

};