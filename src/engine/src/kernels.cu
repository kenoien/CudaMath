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
