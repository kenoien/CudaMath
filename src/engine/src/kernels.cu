#include <cuda_runtime.h>

#include <cstddef>

namespace {

int blocksFor(std::size_t n) {
    const int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    return blocks < 1 ? 1 : blocks;
}

template <typename T>
__global__ void addKernel(T* c, const T* a, const T* b, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

template <typename T>
__global__ void subKernel(T* c, const T* a, const T* b, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

template <typename T>
__global__ void mulKernel(T* c, const T* a, const T* b, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

template <typename T>
__global__ void divKernel(T* c, const T* a, const T* b, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] / b[i];
}

template <typename T>
__global__ void absKernel(T* c, const T* a, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] < 0 ? -a[i] : a[i];
}

__global__ void sqrtKernel(double* c, const double* a, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = sqrt(a[i]);
}

template <typename T>
void launch2(void (*kernel)(T*, const T*, const T*, std::size_t), T* c, const T* a, const T* b, std::size_t n) {
    kernel<<<blocksFor(n), 256>>>(c, a, b, n);
}

template <typename T>
void launch1(void (*kernel)(T*, const T*, std::size_t), T* c, const T* a, std::size_t n) {
    kernel<<<blocksFor(n), 256>>>(c, a, n);
}

}  // namespace

extern "C" void cm_add_f64(double* c, const double* a, const double* b, std::size_t n) { launch2(addKernel<double>, c, a, b, n); }
extern "C" void cm_sub_f64(double* c, const double* a, const double* b, std::size_t n) { launch2(subKernel<double>, c, a, b, n); }
extern "C" void cm_mul_f64(double* c, const double* a, const double* b, std::size_t n) { launch2(mulKernel<double>, c, a, b, n); }
extern "C" void cm_div_f64(double* c, const double* a, const double* b, std::size_t n) { launch2(divKernel<double>, c, a, b, n); }
extern "C" void cm_add_f32(float* c, const float* a, const float* b, std::size_t n) { launch2(addKernel<float>, c, a, b, n); }
extern "C" void cm_sub_f32(float* c, const float* a, const float* b, std::size_t n) { launch2(subKernel<float>, c, a, b, n); }
extern "C" void cm_mul_f32(float* c, const float* a, const float* b, std::size_t n) { launch2(mulKernel<float>, c, a, b, n); }
extern "C" void cm_div_f32(float* c, const float* a, const float* b, std::size_t n) { launch2(divKernel<float>, c, a, b, n); }
extern "C" void cm_add_i32(int* c, const int* a, const int* b, std::size_t n) { launch2(addKernel<int>, c, a, b, n); }
extern "C" void cm_sub_i32(int* c, const int* a, const int* b, std::size_t n) { launch2(subKernel<int>, c, a, b, n); }
extern "C" void cm_mul_i32(int* c, const int* a, const int* b, std::size_t n) { launch2(mulKernel<int>, c, a, b, n); }
extern "C" void cm_add_i64(long long* c, const long long* a, const long long* b, std::size_t n) { launch2(addKernel<long long>, c, a, b, n); }
extern "C" void cm_sub_i64(long long* c, const long long* a, const long long* b, std::size_t n) { launch2(subKernel<long long>, c, a, b, n); }
extern "C" void cm_mul_i64(long long* c, const long long* a, const long long* b, std::size_t n) { launch2(mulKernel<long long>, c, a, b, n); }
extern "C" void cm_abs_f64(double* c, const double* a, std::size_t n) { launch1(absKernel<double>, c, a, n); }
extern "C" void cm_sqrt_f64(double* c, const double* a, std::size_t n) {
    sqrtKernel<<<blocksFor(n), 256>>>(c, a, n);
}

template <typename T>
__device__ T unaryValue(int op, T x);

template <>
__device__ double unaryValue<double>(int op, double x) {
    switch (op) {
        case 0: return x < 0 ? -x : x;
        case 1: return x > 0 ? 1 : (x < 0 ? -1 : 0);
        case 2: return sqrt(x);
        case 3: return exp(x);
        case 4: return log(x);
        case 5: return log2(x);
        case 6: return log10(x);
        case 7: return sin(x);
        case 8: return cos(x);
        case 9: return tan(x);
        case 10: return asin(x);
        case 11: return acos(x);
        case 12: return atan(x);
        case 13: return sinh(x);
        case 14: return cosh(x);
        case 15: return tanh(x);
        case 16: return floor(x);
        case 17: return ceil(x);
        case 18: return round(x);
        default: return trunc(x);
    }
}

template <>
__device__ float unaryValue<float>(int op, float x) {
    switch (op) {
        case 0: return x < 0 ? -x : x;
        case 1: return x > 0 ? 1 : (x < 0 ? -1 : 0);
        case 2: return sqrtf(x);
        case 3: return expf(x);
        case 4: return logf(x);
        case 5: return log2f(x);
        case 6: return log10f(x);
        case 7: return sinf(x);
        case 8: return cosf(x);
        case 9: return tanf(x);
        case 10: return asinf(x);
        case 11: return acosf(x);
        case 12: return atanf(x);
        case 13: return sinhf(x);
        case 14: return coshf(x);
        case 15: return tanhf(x);
        case 16: return floorf(x);
        case 17: return ceilf(x);
        case 18: return roundf(x);
        default: return truncf(x);
    }
}

template <typename T>
__global__ void unaryKernel(int op, T* c, const T* a, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = unaryValue<T>(op, a[i]);
}

__global__ void unaryIntKernel(int op, int* c, const int* a, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int x = a[i];
    c[i] = op == 0 ? (x < 0 ? -x : x) : (x > 0 ? 1 : (x < 0 ? -1 : 0));
}

__global__ void unaryLongKernel(int op, long long* c, const long long* a, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long x = a[i];
    c[i] = op == 0 ? (x < 0 ? -x : x) : (x > 0 ? 1 : (x < 0 ? -1 : 0));
}

template <typename T>
__device__ T binaryValue(int op, T a, T b) {
    if (op == 0) return pow(a, b);
    if (op == 1) return atan2(a, b);
    if (op == 2) {
        if (b == T(0)) return T(0);
        T q = floor(a / b);
        return a - b * q;
    }
    if (b == T(0)) return T(0);
    return a - b * trunc(a / b);
}

template <typename T>
__global__ void binaryMathKernel(int op, T* c, const T* a, const T* b, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = binaryValue<T>(op, a[i], b[i]);
}

template <typename T>
__global__ void colReduceKernel(int op, T* out, const T* in, int rows, int cols) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= cols) return;
    if (rows <= 0) {
        out[col] = op == 1 ? T(1) : T(0);
        return;
    }
    T acc = in[col * rows];
    if (op == 0 || op == 1) acc = op == 0 ? T(0) : T(1);
    for (int row = 0; row < rows; ++row) {
        const T value = in[col * rows + row];
        if (op == 0) acc += value;
        else if (op == 1) acc *= value;
        else if (op == 2) acc = value < acc ? value : acc;
        else if (op == 3) acc = value > acc ? value : acc;
        else if (op == 4) acc = (acc != T(0) || value != T(0)) ? T(1) : T(0);
        else acc = (acc != T(0) && value != T(0)) ? T(1) : T(0);
    }
    if (op == 4 || op == 5) acc = acc != T(0) ? T(1) : T(0);
    out[col] = acc;
}

template <typename T>
__global__ void rowReduceKernel(int op, T* out, const T* in, int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    if (cols <= 0) {
        out[row] = op == 1 ? T(1) : T(0);
        return;
    }
    T acc = op == 0 ? T(0) : (op == 1 ? T(1) : in[row]);
    for (int col = 0; col < cols; ++col) {
        const T value = in[col * rows + row];
        if (op == 0) acc += value;
        else if (op == 1) acc *= value;
        else if (op == 2) acc = col == 0 || value < acc ? value : acc;
        else if (op == 3) acc = col == 0 || value > acc ? value : acc;
        else if (op == 4) acc = (acc != T(0) || value != T(0)) ? T(1) : T(0);
        else acc = (acc != T(0) && value != T(0)) ? T(1) : T(0);
    }
    if (op == 4 || op == 5) acc = acc != T(0) ? T(1) : T(0);
    out[row] = acc;
}

__global__ void crossKernel(double* c, const double* a, const double* b) {
    const int i = static_cast<int>(threadIdx.x);
    if (i == 0) c[0] = a[1] * b[2] - a[2] * b[1];
    if (i == 1) c[1] = a[2] * b[0] - a[0] * b[2];
    if (i == 2) c[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename T>
void launchUnary(int op, T* c, const T* a, std::size_t n) {
    unaryKernel<T><<<blocksFor(n), 256>>>(op, c, a, n);
}

template <typename T>
void launchBinary(int op, T* c, const T* a, const T* b, std::size_t n) {
    binaryMathKernel<T><<<blocksFor(n), 256>>>(op, c, a, b, n);
}

template <typename T>
void launchCol(int op, T* out, const T* in, int rows, int cols) {
    colReduceKernel<T><<<blocksFor(static_cast<std::size_t>(cols)), 256>>>(op, out, in, rows, cols);
}

template <typename T>
void launchRow(int op, T* out, const T* in, int rows, int cols) {
    rowReduceKernel<T><<<blocksFor(static_cast<std::size_t>(rows)), 256>>>(op, out, in, rows, cols);
}

extern "C" void cm_unary_f64(int op, double* c, const double* a, std::size_t n) { launchUnary(op, c, a, n); }
extern "C" void cm_unary_f32(int op, float* c, const float* a, std::size_t n) { launchUnary(op, c, a, n); }
extern "C" void cm_unary_i32(int op, int* c, const int* a, std::size_t n) {
    unaryIntKernel<<<blocksFor(n), 256>>>(op, c, a, n);
}
extern "C" void cm_unary_i64(int op, long long* c, const long long* a, std::size_t n) {
    unaryLongKernel<<<blocksFor(n), 256>>>(op, c, a, n);
}
extern "C" void cm_binary_f64(int op, double* c, const double* a, const double* b, std::size_t n) { launchBinary(op, c, a, b, n); }
extern "C" void cm_binary_f32(int op, float* c, const float* a, const float* b, std::size_t n) { launchBinary(op, c, a, b, n); }
extern "C" void cm_col_f64(int op, double* out, const double* in, int rows, int cols) { launchCol(op, out, in, rows, cols); }
extern "C" void cm_col_f32(int op, float* out, const float* in, int rows, int cols) { launchCol(op, out, in, rows, cols); }
extern "C" void cm_col_i32(int op, int* out, const int* in, int rows, int cols) { launchCol(op, out, in, rows, cols); }
extern "C" void cm_col_i64(int op, long long* out, const long long* in, int rows, int cols) { launchCol(op, out, in, rows, cols); }
extern "C" void cm_row_f64(int op, double* out, const double* in, int rows, int cols) { launchRow(op, out, in, rows, cols); }
extern "C" void cm_row_f32(int op, float* out, const float* in, int rows, int cols) { launchRow(op, out, in, rows, cols); }
extern "C" void cm_row_i32(int op, int* out, const int* in, int rows, int cols) { launchRow(op, out, in, rows, cols); }
extern "C" void cm_row_i64(int op, long long* out, const long long* in, int rows, int cols) { launchRow(op, out, in, rows, cols); }
extern "C" void cm_cross_f64(double* c, const double* a, const double* b) { crossKernel<<<1, 3>>>(c, a, b); }
