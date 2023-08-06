#pragma once

#include <memory>

template<typename T>
/**
 * @brief  Matrix1d is a simple wrapper class for simple arrays. Originally, I used a std::vector in most places of the code, but due to efficiency reasons (e.g. converting a numpy array from python to a std::vector usually involves a copy) I decided to use c-like arrays instead. Hence, the capabilities of this class are more limited compared to std::vectors since it does not support assingment operators or copy constructors. Moreover, dynamic re-sizeing (e.g. via push_back) is also not supported. Internally this class uses a unique_ptr to manage the underlying data. For compatibility to the other matrix classes (see below) and for compatibility to the way pybind is handling numpy arrays, this class can explicitly refuse to own the data pointed to by the unique_ptr. In this case, the data is not removed and kept alive beyond the lifetime of the objects of this class. This is necessary for the Python binding and useful for returning light-weight "views" on the data, e.g. return a row of a matrix2d as matrix1d etc. However, if the data is not owned, then it is not guaranteed that access to it will be valid at any point in time since data can be removed externally. This is intendend behaviour. Any implicit copies are avoided to maximize performance. If you require a copy of your data, then copy it manually. 
 * 
 * TODO: Maybe a shared_ptr makes more sense for this class?
 * @note   
 * @retval None
 */
class matrix1d {

public:
    // Note: I guess it is debateable if it is good or bad practice to use public members, especially if this class assume a certain state for the members. I found that private members did not offer much of a benefit except that we guarantee that they are not changed, but convoluted the interface and made the Python bindings a bit more tricky. 

    /**
     * @brief  The dimensionality (=number of data items) in this vector.
     * @note   
     * @retval None
     */
    unsigned int dim;

    /**
     * @brief  A unique ptr that points to an array.
     * @note   
     * @retval None
     */
    std::unique_ptr<T[]> data;

    /**
     * @brief  If true, then upon deleting the object, the unique_ptr also deletes its associated memory. If false, the associated memory is kept alive (i.e. if this is mananged outside the C++ code by Python / Numpy)
     * @note   
     * @retval None
     */
    bool has_ownership;

    
    /**
     * @brief  Default c'tor for this class. This is required for the type_caster of pybind in "PythonBindings.cpp"
     * @note   
     * @retval The newly generated class object
     */
    matrix1d() : dim(0), data(nullptr), has_ownership(false) {}

    /**
     * @brief  Creates a new vector with dim entries from the given src, without copying the underlying data. The pointer to src is simply moved into the unique_ptr. If take_ownership is set to true, then this object takes ownership of the memory and deletes it when this object is deleted.
     * @note   
     * @param  dim: The dimensionality (=number of data items) in this vector.
     * @param  src: A pointer to the source data. 
     * @param  take_ownership: If true, then this class starts to manage the given source data, i.e. deletes it upon deletion of this object. 
     * @retval The newly generated class object
     */
    matrix1d(unsigned int dim, T * const src, bool take_ownership = true) 
    : dim(dim), data(src), has_ownership(take_ownership) {}

    /**
     * @brief  Creates a new vector with dim entries. Entries are not initialized. Memory for the vector is allocated on the heap and is owned by this object. Hence, it is automatically freed if this object is deleted.
     * @note   
     * @param  dim: The dimensionality (=number of data items) in this vector.
     * @retval The newly generated class object
     */
    matrix1d(unsigned int dim) 
    : dim(dim), data(new T[dim]), has_ownership(true) {}

    /**
     * @brief  Move constructor.
     * @note   
     * @param  a: The other object.
     * @retval The newly generated class object
     */
    matrix1d( matrix1d&& a ) : dim(a.dim), data( std::move( a.data ) ), has_ownership(a.has_ownership) {}

    matrix1d& operator=(matrix1d&& other) {
        dim = other.dim;
        data = std::move(other.data);
        has_ownership = other.has_ownership;
        return *this;
    }

    // Prevent copy and assignment operations. This leads to more difficulties than it solves problems. 
    matrix1d( matrix1d& a ) = delete;
    matrix1d( const matrix1d& a ) = delete;
    matrix1d& operator=(const matrix1d& other) = delete;

    ~matrix1d() {
        // Make sure we only delete the underlying data if we actually own it. 
        if (!has_ownership) {
            data.release();
        }
    }

    /**
     * @brief  Returns the beginning of the underlying continous data structure. Makes it easier to use the STL with this class. 
     * @note   
     * @retval Pointer to the beginning of the underlying data structure 
     */
    auto begin() const {
        return &data[0];
    }

    /**
     * @brief  Returns the beginning of the underlying continous data structure. Makes it easier to use the STL with this class. 
     * @note   
     * @retval Pointer to the beginning of the underlying data structure 
     */
    auto begin() {
        return &data.get()[0];
    }

    /**
     * @brief  Returns the end of the underlying continous data structure. Makes it easier to use the STL with this class. 
     * @note   
     * @retval Pointer to the end of the underlying data structure 
     */
    auto end() const {
        return &data[dim];
    }

    /**
     * @brief  Returns the end of the underlying continous data structure. Makes it easier to use the STL with this class. 
     * @note   
     * @retval Pointer to the end of the underlying data structure 
     */
    auto end() {
        return &data.get()[dim];
    }

    /**
     * @brief  Gives access to the i-th element of this 0-based vector. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The index. 
     * @retval The element at the i-th position of this vector. 
     */
    T& operator()(unsigned int i) {
        return data[i];
    }

    /**
     * @brief  Gives access to the i-th element of this 0-based vector. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The index. 
     * @retval The element at the i-th position of this vector. 
     */
    T operator()(unsigned int i) const {
        return data.get()[i];
    }

    unsigned int num_bytes() const {
        if (has_ownership) {
            return sizeof(*this) + sizeof(T)*dim;
        } else {
            return sizeof(*this);
        }
    }
};

template<typename T>
/**
 * @brief  matrix2d is a simple wrapper class for simple arrays. Originally, I used a std::vector<std::vetor<T>> in most places of the code, but due to efficiency reasons I decided to a single c-like array instead. Hence, the capabilities of this class are more limited compared to std::vectors since it does not support assingment operators or copy constructors. Moreover, dynamic re-sizeing (e.g. via push_back) is also not supported. Internally this class uses a unique_ptr to manage the underlying data. For compatibility to the other matrix classes and for compatibility to the way pybind is handling numpy arrays, this class can explicitly refuse to own the data pointed to by the unique_ptr. In this case, the data is not removed and kept alive beyond the lifetime of the objects of this class. This is necessary for the Python binding and useful for returning light-weight "views" on the data, e.g. return a row of a matrix2d as matrix1d etc. However, if the data is not owned, then it is not guaranteed that access to it will be valid at any point in time since data can be removed externally. This is intendend behaviour. Any implicit copies are avoided to maximize performance. If you require a copy of your data, then copy it manually. 
 * @note   
 * @retval None
 */
class matrix2d {
public:
    // Note: I guess it is debateable if it is good or bad practice to use public members, especially if this class assume a certain state for the members. I found that private members did not offer much of a benefit except that we guarantee that they are not changed, but convoluted the interface and made the Python bindings a bit more tricky. 

    /**
     * @brief  Number of rows in this matrix
     * @note   
     * @retval None
     */
    unsigned int rows;

    /**
     * @brief  Number of columns in this matrix
     * @note   
     * @retval None
     */
    unsigned int cols;

    /**
     * @brief  A unique ptr that points to an array.
     * @note   
     * @retval None
     */
    std::unique_ptr<T[]> data;

    /**
     * @brief  If true, then upon deleting the object, the unique_ptr also deletes its associated memory. If false, the associated memory is kept alive (i.e. if this is mananged outside the C++ code by Python / Numpy)
     * @note   
     * @retval None
     */
    bool has_ownership;

    /**
    * @brief  Default c'tor for this class. This is required for the type_caster of pybind in "PythonBindings.cpp"
    * @note   
    * @retval The newly generated class object
    */
    matrix2d() : rows(0), cols(0), data(nullptr), has_ownership(false) {}

    /**
    * @brief  Creates a new vector with dim entries from the given src, without copying the underlying data. The pointer to src is simply moved into the unique_ptr. If take_ownership is set to true, then this object takes ownership of the memory and deletes it when this object is deleted.
    * @note   
    * @param  rows: The number of rows in this matrix
    * @param  columns: The number of columns in this matrix
    * @param  src: A pointer to the source data. 
    * @param  take_ownership: If true, then this class starts to manage the given source data, i.e. deletes it upon deletion of this object. 
    * @retval The newly generated class object
    */
    matrix2d(unsigned int rows, unsigned int columns, T * const src, bool take_ownership = true) 
    : rows(rows), cols(columns), data(src), has_ownership(take_ownership) {
    }

    /**
    * @brief  Creates a new vector with dim entries. Entries are not initialized. Memory for the vector is allocated on the heap and is owned by this object. Hence, it is automatically freed if this object is deleted.
    * @note   
    * @param  rows: The number of rows in this matrix
    * @param  columns: The number of columns in this matrix
    * @retval The newly generated class object
    */
    matrix2d(unsigned int rows, unsigned int columns) 
    : rows(rows), cols(columns), data(new T[rows * columns]), has_ownership(true) { }

    /**
    * @brief  Move constructor.
    * @note   
    * @param  a: The other object.
    * @retval The newly generated class object
    */
    matrix2d( matrix2d&& a ) : rows(a.rows), cols(a.cols), data( std::move( a.data ) ), has_ownership(a.has_ownership){}

    // Prevent copy and assignment operations. This leads to more difficulties than it solves problems. 
    matrix2d( matrix2d& a ) = delete;
    matrix2d( const matrix2d& a ) = delete;
    matrix2d& operator=(const matrix2d& other) = delete;
    matrix2d& operator=(matrix2d&& other) {
        rows = other.rows;
        cols = other.cols;
        data = std::move(other.data);
        has_ownership = other.has_ownership;
        return *this;
    }

    ~matrix2d() {
        // Make sure we only delete the underlying data if we actually own it. 
        if (!has_ownership) {
            data.release();
        }
    }

    /**
    * @brief  Returns the beginning of the underlying continous data structure. Makes it easier to use the STL with this class. 
    * @note   
    * @retval Pointer to the beginning of the underlying data structure 
    */
    auto begin() const {
        return &data[0];
    }

    /**
    * @brief  Returns the beginning of the underlying continous data structure. Makes it easier to use the STL with this class. 
    * @note   
    * @retval Pointer to the beginning of the underlying data structure 
    */
    auto begin() {
        return &data.get()[0];
    }

    /**
    * @brief  Returns the end of the underlying continous data structure. Makes it easier to use the STL with this class. 
    * @note   
    * @retval Pointer to the end of the underlying data structure 
    */
    auto end() const {
        return &data[cols * rows];
    }

    /**
    * @brief  Returns the end of the underlying continous data structure. Makes it easier to use the STL with this class. 
    * @note   
    * @retval Pointer to the end of the underlying data structure 
    */
    auto end() {
        return &data.get()[cols * rows];
    }

    /**
     * @brief  Returns the i-th row of the 0-based matrix as a matrix1d<T>. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The i-th row.
     * @retval A new object of type matri1d<T> that points to the row in the underlying data structure. This object does not own any data, so the original matrix remains alive when it is removed. However, the new object also becomes invalid as soon as the original matrix is removed. No data is copied at any point.
     */
    matrix1d<T> operator()(unsigned int i) const {
        return matrix1d<T>(cols,&data.get()[i * cols], false);
    }

    /**
     * @brief  Gives access to the (i,j) element in this matrix where i and j start with 0. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The i-th row. 
     * @param  j: The j-th column. 
     * @retval The element at the (i,j) element in this matrix
     */
    T& operator()(unsigned int i, unsigned int j) {
        return data.get()[i * cols + j];
    }

    /**
     * @brief  Gives access to the (i,j) element in this matrix where i and j start with 0. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The i-th row. 
     * @param  j: The j-th column. 
     * @retval The element at the (i,j) element in this matrix
     */
    T operator()(unsigned int i, unsigned int j) const {
        return data.get()[i * cols + j];
    }

    unsigned int num_bytes() const {
        if (has_ownership) {
            return sizeof(*this) + sizeof(T)*rows*cols;
        } else {
            return sizeof(*this);
        }
    }
};

template<typename T>
/**
 * @brief  matrix3d is a simple wrapper class for simple arrays. Originally, I used a std::vector<std::vetor<std::vector<T>>> in most places of the code, but due to efficiency reasons I decided to use a single c-like array instead. Hence, the capabilities of this class are more limited compared to std::vectors since it does not support assingment operators or copy constructors. Moreover, dynamic re-sizeing (e.g. via push_back) is also not supported. Internally this class uses a unique_ptr to manage the underlying data. For compatibility to the other matrix classes and for compatibility to the way pybind is handling numpy arrays, this class can explicitly refuse to own the data pointed to by the unique_ptr. In this case, the data is not removed and kept alive beyond the lifetime of the objects of this class. This is necessary for the Python binding and useful for returning light-weight "views" on the data, e.g. return a row of a matrix2d as matrix1d etc. However, if the data is not owned, then it is not guaranteed that access to it will be valid at any point in time since data can be removed externally. This is intendend behaviour. Any implicit copies are avoided to maximize performance. If you require a copy of your data, then copy it manually. 
 * @note   
 * @retval None
 */
class matrix3d {

public:
    // Note: I guess it is debateable if it is good or bad practice to use public members, especially if this class assume a certain state for the members. I found that private members did not offer much of a benefit except that we guarantee that they are not changed, but convoluted the interface and made the Python bindings a bit more tricky. 

    /**
     * @brief  The x-dim in this tensor
     * @note   
     * @retval None
     */
    unsigned int nx;

    /**
     * @brief  The y-dim in this tensor
     * @note   
     * @retval None
     */
    unsigned int ny;

    /**
     * @brief  The z-dim in this tensor
     * @note   
     * @retval None
     */
    unsigned int nz;

     /**
     * @brief  A unique ptr that points to an array.
     * @note   
     * @retval None
     */
    std::unique_ptr<T[]> data;

    /**
     * @brief  If true, then upon deleting the object, the unique_ptr also deletes its associated memory. If false, the associated memory is kept alive (i.e. if this is mananged outside the C++ code by Python / Numpy)
     * @note   
     * @retval None
     */
    bool has_ownership;

    /**
    * @brief  Default c'tor for this class. 
    * @note   
    * @retval The newly generated class object
    */
    matrix3d() : nx(0), ny(0), nz(0), data(nullptr), has_ownership(false) {}

    /**
    * @brief  Creates a new tensor with nx/ny/nz dimensions from the given src, without copying the underlying data. The pointer to src is simply moved into the unique_ptr. If take_ownership is set to true, then this object takes ownership of the memory and deletes it when this object is deleted.
    * @note   
    * @param  nx: The x-dim of this tensor
    * @param  ny: The y-dim of this tensor
    * @param  nz: The z-dim of this tensor
    * @param  src: A pointer to the source data. 
    * @param  take_ownership: If true, then this class starts to manage the given source data, i.e. deletes it upon deletion of this object. 
    * @retval The newly generated class object
    */
    matrix3d(unsigned int nx, unsigned int ny, unsigned int nz, T * const src, bool take_ownership = true) 
    : nx(nx), ny(ny), nz(nz), data(src), has_ownership(take_ownership) {}

     /**
    * @brief  Creates a new tensor with nx/ny/nz dimensions. Entries are not initialized. Memory for the tensor is allocated on the heap and is owned by this object. Hence, it is automatically freed if this object is deleted.
    * @note   
    * @param  nx: The x-dim of this tensor
    * @param  ny: The y-dim of this tensor
    * @param  nz: The z-dim of this tensor
    * @retval The newly generated class object
    */
    matrix3d(unsigned int nx, unsigned int ny, unsigned int nz) 
    : nx(nx), ny(ny), nz(nz), data(new T[nx * ny * nz]), has_ownership(true) {}

    /**
    * @brief  Move constructor.
    * @note   
    * @param  a: The other object.
    * @retval The newly generated class object
    */
    matrix3d( matrix3d&& a ) : nx(a.nx), ny(a.ny), nz(a.nz), data( std::move( a.data ) ) {}

    // Prevent copy and assignment operations. This leads to more difficulties than it solves problems. 
    matrix3d( matrix3d& a ) = delete;
    matrix3d( const matrix3d& a ) = delete;
    matrix3d& operator=(const matrix3d& other) = delete;
    //matrix3d& operator=(const matrix3d&& other) = delete;
    
    matrix3d& operator=(const matrix3d&& other) {
        nx = other.nx;
        ny = other.ny;
        nz = other.nz;
        data = std::move(other.data);
        has_ownership = other.has_ownership;
        return *this;
    }

    ~matrix3d() {
        if (!has_ownership) {
            // Make sure we only delete the underlying data if we actually own it. 
            data.release();
        }
    }

    /**
    * @brief  Returns the beginning of the underlying continous data structure. Makes it easier to use the STL with this class. 
    * @note   
    * @retval Pointer to the beginning of the underlying data structure 
    */
    auto begin() const {
        return &data[0];
    }

    /**
    * @brief  Returns the beginning of the underlying continous data structure. Makes it easier to use the STL with this class. 
    * @note   
    * @retval Pointer to the beginning of the underlying data structure 
    */
    auto begin() {
        return &data[0];
    }

    /**
    * @brief  Returns the end of the underlying continous data structure. Makes it easier to use the STL with this class. 
    * @note   
    * @retval Pointer to the end of the underlying data structure 
    */
    auto end() const {
        return &data[nx * ny * nz];
    }

    /**
    * @brief  Returns the end of the underlying continous data structure. Makes it easier to use the STL with this class. 
    * @note   
    * @retval Pointer to the end of the underlying data structure 
    */
    auto end() {
        return &data[nx * ny * nz];
    }

    /**
     * @brief  Returns the i-th sub matrix (0-based indexing) as a matrix2d<T>. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The i-submatrix.
     * @retval A new object of type matri2d<T> that points to the matrix in the underlying data structure. This object does not own any data, so the original tensor remains alive when it is removed. However, the new object also becomes invalid as soon as the original tensor is removed. No data is copied at any point.
     */
    matrix2d<T> operator()(unsigned int i) {
        matrix2d<T> tmp(ny,nz,&data.get()[i * (ny * nz)], false);

        return tmp;
    }

    /**
     * @brief  Returns the (i,j)-th row (0-based indexing) as a matrix1d<T>. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The i-th coordinate
     * @param  j: The j-to coordinate.
     * @retval A new object of type matri1d<T> that points to the row in the underlying data structure. This object does not own any data, so the original tensor remains alive when it is removed. However, the new object also becomes invalid as soon as the original tensor is removed. No data is copied at any point.
     */
    matrix1d<T> operator()(unsigned int i, unsigned j) {
        matrix1d<T> tmp(ny,nz,&data[i * (ny * nz) + j * nz], false);

        return tmp;
    }

    /**
     * @brief  Gives access to the (i,j,k) element in this tensor where i, j and k start with 0. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The i-th coordinate. 
     * @param  j: The j-th coordinate. 
     * @param  k: The j-th coordinate. 
     * @retval The element at the (i,j,k) element in this tensor
     */
    T& operator()(unsigned int i, unsigned int j, unsigned int k) {
        return &data[k + j*nz + i*ny*nz];
    }

    /**
     * @brief  Gives access to the (i,j,k) element in this tensor where i, j and k start with 0. For maximum performance no boundary checks are performed!
     * @note   
     * @param  i: The i-th coordinate. 
     * @param  j: The j-th coordinate. 
     * @param  k: The j-th coordinate. 
     * @retval The element at the (i,j,k) element in this tensor
     */
    T operator()(unsigned int i, unsigned int j, unsigned int k) const {
        return data.get()[k + j*nz + i*ny*nz];
    }

    unsigned int num_bytes() const {
        if (has_ownership) {
            return sizeof(*this) + sizeof(T)*nx*ny*nz;
        } else {
            return sizeof(*this);
        }
    }
};