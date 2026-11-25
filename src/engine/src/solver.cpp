#include "cudamath/ops.hpp"

#include "cudamath/engine.hpp"

#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cudamath {
namespace {

void sync() {
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

cusolverDnHandle_t handle() {
    return reinterpret_cast<cusolverDnHandle_t>(Engine::instance().solver());
}

template <typename T>
struct Device {
    T* data{nullptr};
    explicit Device(std::size_t count) {
        if (count == 0) return;
        const cudaError_t status = cudaMallocManaged(&data, count * sizeof(T));
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    ~Device() {
        if (data != nullptr) cudaFree(data);
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

Value copyDouble(const Value& value) {
    if (value.isString) throw std::runtime_error("solver requires a numeric array");
    Value out = Value::matrix(DType::F64, value.rows, value.cols);
    for (std::int64_t i = 0; i < value.numel(); ++i) out.set(i, value.get(i));
    return out;
}

int readInfo(int* info) {
    sync();
    return *info;
}

struct LuWork {
    Value matrix;
    std::vector<int> pivot;
};

LuWork decompose(const Value& value) {
    if (value.rows > 2147483647 || value.cols > 2147483647) throw std::runtime_error("matrix is too large");
    LuWork work;
    work.matrix = copyDouble(value);
    const int m = static_cast<int>(value.rows);
    const int n = static_cast<int>(value.cols);
    const int k = std::min(m, n);
    int lwork = 0;
    auto* data = static_cast<double*>(work.matrix.data());
    if (cusolverDnDgetrf_bufferSize(handle(), m, n, data, m, &lwork) != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER getrf failed");
    }
    Device<double> buffer(static_cast<std::size_t>(std::max(lwork, 1)));
    Device<int> ipiv(static_cast<std::size_t>(std::max(k, 1)));
    Device<int> info(1);
    if (cusolverDnDgetrf(handle(), m, n, data, m, buffer.data, ipiv.data, info.data) != CUSOLVER_STATUS_SUCCESS || readInfo(info.data) != 0) {
        throw std::runtime_error("cuSOLVER getrf failed");
    }
    work.pivot.assign(ipiv.data, ipiv.data + k);
    return work;
}

Value identity(std::int64_t n) {
    Value out = Value::matrix(DType::F64, n, n);
    for (std::int64_t i = 0; i < n; ++i) out.set(i * n + i, 1);
    return out;
}

Value solve(const LuWork& lu, Value right) {
    const int n = static_cast<int>(lu.matrix.rows);
    const int nrhs = static_cast<int>(right.cols);
    if (right.rows != lu.matrix.rows) throw std::runtime_error("backslash dimensions must agree");
    Device<int> ipiv(lu.pivot.size());
    for (std::size_t i = 0; i < lu.pivot.size(); ++i) ipiv.data[i] = lu.pivot[i];
    Device<int> info(1);
    if (cusolverDnDgetrs(handle(), CUBLAS_OP_N, n, nrhs, static_cast<double*>(lu.matrix.data()), n, ipiv.data,
            static_cast<double*>(right.data()), static_cast<int>(right.rows), info.data) != CUSOLVER_STATUS_SUCCESS ||
        readInfo(info.data) != 0) {
        throw std::runtime_error("cuSOLVER getrs failed");
    }
    return right;
}

Value permutation(const std::vector<int>& pivot, std::int64_t n) {
    std::vector<int> rows(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) rows[static_cast<std::size_t>(i)] = i;
    for (int i = 0; i < static_cast<int>(pivot.size()) && i < n; ++i) {
        std::swap(rows[static_cast<std::size_t>(i)], rows[static_cast<std::size_t>(pivot[static_cast<std::size_t>(i)] - 1)]);
    }
    Value out = Value::matrix(DType::F64, n, n);
    for (std::int64_t row = 0; row < n; ++row) out.set(static_cast<std::int64_t>(rows[static_cast<std::size_t>(row)]) * n + row, 1);
    return out;
}

Value finish(const Value& value, DType type) {
    if (type == DType::F32) return convertTo(value, DType::F32);
    return value;
}

bool symmetric(const Value& value) {
    if (value.rows != value.cols) return false;
    for (std::int64_t c = 0; c < value.cols; ++c) {
        for (std::int64_t r = c + 1; r < value.rows; ++r) {
            if (std::abs(value.get(c * value.rows + r) - value.get(r * value.rows + c)) > 1e-10) return false;
        }
    }
    return true;
}

}  // namespace

std::vector<Value> factorLu(const Value& value, int outputs) {
    const DType type = value.type == DType::F32 ? DType::F32 : DType::F64;
    LuWork lu = decompose(value);
    if (outputs <= 1) return {finish(lu.matrix, type)};
    const std::int64_t m = value.rows;
    const std::int64_t n = value.cols;
    Value lower = Value::matrix(DType::F64, m, std::min(m, n));
    Value upper = Value::matrix(DType::F64, std::min(m, n), n);
    for (std::int64_t c = 0; c < n; ++c) {
        for (std::int64_t r = 0; r < m; ++r) {
            const double entry = lu.matrix.get(c * m + r);
            if (r > c && c < lower.cols) lower.set(c * lower.rows + r, entry);
            if (r == c && r < lower.rows && c < lower.cols) lower.set(c * lower.rows + r, 1);
            if (r <= c && r < upper.rows) upper.set(c * upper.rows + r, entry);
        }
    }
    Value perm = m == n ? permutation(lu.pivot, n) : identity(std::max(m, n));
    if (m != n) perm = permutation(lu.pivot, m);
    return {finish(lower, type), finish(upper, type), finish(perm, type)};
}

std::vector<Value> factorQr(const Value& value, int outputs) {
    const DType type = value.type == DType::F32 ? DType::F32 : DType::F64;
    Value factor = copyDouble(value);
    const int m = static_cast<int>(value.rows);
    const int n = static_cast<int>(value.cols);
    const int k = std::min(m, n);
    int lwork = 0;
    auto* data = static_cast<double*>(factor.data());
    if (cusolverDnDgeqrf_bufferSize(handle(), m, n, data, m, &lwork) != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER geqrf failed");
    }
    Device<double> tau(static_cast<std::size_t>(std::max(k, 1)));
    Device<double> buffer(static_cast<std::size_t>(std::max(lwork, 1)));
    Device<int> info(1);
    if (cusolverDnDgeqrf(handle(), m, n, data, m, tau.data, buffer.data, lwork, info.data) != CUSOLVER_STATUS_SUCCESS || readInfo(info.data) != 0) {
        throw std::runtime_error("cuSOLVER geqrf failed");
    }
    Value upper = Value::matrix(DType::F64, m, n);
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r <= c && r < m; ++r) upper.set(c * m + r, factor.get(c * m + r));
    }
    if (outputs <= 1) return {finish(upper, type)};
    Value q = Value::matrix(DType::F64, m, m);
    for (int c = 0; c < n && c < m; ++c) {
        for (int r = 0; r < m; ++r) q.set(c * m + r, factor.get(c * m + r));
    }
    int qwork = 0;
    if (cusolverDnDorgqr_bufferSize(handle(), m, m, k, static_cast<double*>(q.data()), m, tau.data, &qwork) != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER orgqr failed");
    }
    Device<double> qbuffer(static_cast<std::size_t>(std::max(qwork, 1)));
    if (cusolverDnDorgqr(handle(), m, m, k, static_cast<double*>(q.data()), m, tau.data, qbuffer.data, qwork, info.data) != CUSOLVER_STATUS_SUCCESS ||
        readInfo(info.data) != 0) {
        throw std::runtime_error("cuSOLVER orgqr failed");
    }
    return {finish(q, type), finish(upper, type)};
}

std::vector<Value> factorSvd(const Value& value, int outputs) {
    const DType type = value.type == DType::F32 ? DType::F32 : DType::F64;
    Value factor = copyDouble(value);
    const int m = static_cast<int>(value.rows);
    const int n = static_cast<int>(value.cols);
    const int k = std::min(m, n);
    int lwork = 0;
    if (cusolverDnDgesvd_bufferSize(handle(), m, n, &lwork) != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER gesvd failed");
    }
    Value singular = Value::matrix(DType::F64, k, 1);
    Value u = Value::matrix(DType::F64, m, m);
    Value vt = Value::matrix(DType::F64, n, n);
    Device<double> buffer(static_cast<std::size_t>(std::max(lwork, 1)));
    Device<double> rwork(static_cast<std::size_t>(std::max(1, std::min(m, n) - 1)));
    Device<int> info(1);
    if (cusolverDnDgesvd(handle(), 'A', 'A', m, n, static_cast<double*>(factor.data()), m, static_cast<double*>(singular.data()),
            static_cast<double*>(u.data()), m, static_cast<double*>(vt.data()), n, buffer.data, lwork, rwork.data, info.data) !=
            CUSOLVER_STATUS_SUCCESS ||
        readInfo(info.data) != 0) {
        throw std::runtime_error("cuSOLVER gesvd failed");
    }
    if (outputs <= 1) return {finish(singular, type)};
    Value s = Value::matrix(DType::F64, m, n);
    for (int i = 0; i < k; ++i) s.set(i * m + i, singular.get(i));
    Value v = transpose(vt);
    return {finish(u, type), finish(s, type), finish(v, type)};
}

std::vector<Value> factorEig(const Value& value, int outputs) {
    if (value.rows != value.cols) throw std::runtime_error("eig requires a square matrix");
    const DType type = value.type == DType::F32 ? DType::F32 : DType::F64;
    const int n = static_cast<int>(value.rows);
    Value factor = copyDouble(value);
    Value values = Value::matrix(DType::F64, n, 1);
    if (symmetric(value)) {
        int lwork = 0;
        if (cusolverDnDsyevd_bufferSize(handle(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, static_cast<double*>(factor.data()), n,
                static_cast<double*>(values.data()), &lwork) != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cuSOLVER syevd failed");
        }
        Device<double> buffer(static_cast<std::size_t>(std::max(lwork, 1)));
        Device<int> info(1);
        if (cusolverDnDsyevd(handle(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, static_cast<double*>(factor.data()), n,
                static_cast<double*>(values.data()), buffer.data, lwork, info.data) != CUSOLVER_STATUS_SUCCESS ||
            readInfo(info.data) != 0) {
            throw std::runtime_error("cuSOLVER syevd failed");
        }
        if (outputs <= 1) return {finish(values, type)};
        Value diag = Value::matrix(DType::F64, n, n);
        for (int i = 0; i < n; ++i) diag.set(i * n + i, values.get(i));
        return {finish(factor, type), finish(diag, type)};
    }
    cusolverDnParams_t params = nullptr;
    if (cusolverDnCreateParams(&params) != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER geev failed");
    Value vectors = Value::matrix(DType::F64, n, n);
    Device<cuDoubleComplex> eigenvalues(static_cast<std::size_t>(n));
    size_t deviceBytes = 0;
    size_t hostBytes = 0;
    if (cusolverDnXgeev_bufferSize(handle(), params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR, n, CUDA_R_64F, factor.data(), n, CUDA_C_64F,
            eigenvalues.data, CUDA_R_64F, nullptr, 1, CUDA_R_64F, vectors.data(), n, CUDA_R_64F, &deviceBytes, &hostBytes) != CUSOLVER_STATUS_SUCCESS) {
        cusolverDnDestroyParams(params);
        throw std::runtime_error("cuSOLVER geev failed");
    }
    Device<char> deviceBuffer(std::max<std::size_t>(deviceBytes, 1));
    std::vector<char> hostBuffer(std::max<std::size_t>(hostBytes, 1));
    Device<int> info(1);
    const cusolverStatus_t status = cusolverDnXgeev(handle(), params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR, n, CUDA_R_64F, factor.data(), n,
        CUDA_C_64F, eigenvalues.data, CUDA_R_64F, nullptr, 1, CUDA_R_64F, vectors.data(), n, CUDA_R_64F, deviceBuffer.data, deviceBytes, hostBuffer.data(),
        hostBytes, info.data);
    cusolverDnDestroyParams(params);
    if (status != CUSOLVER_STATUS_SUCCESS || readInfo(info.data) != 0) throw std::runtime_error("cuSOLVER geev failed");
    for (int i = 0; i < n; ++i) {
        if (std::abs(eigenvalues.data[i].y) > 1e-8) throw std::runtime_error("complex eigenvalue");
        values.set(i, eigenvalues.data[i].x);
    }
    if (outputs <= 1) return {finish(values, type)};
    Value diag = Value::matrix(DType::F64, n, n);
    for (int i = 0; i < n; ++i) diag.set(i * n + i, values.get(i));
    return {finish(vectors, type), finish(diag, type)};
}

Value chol(const Value& value) {
    if (value.rows != value.cols) throw std::runtime_error("chol requires a square matrix");
    const DType type = value.type == DType::F32 ? DType::F32 : DType::F64;
    Value factor = copyDouble(value);
    const int n = static_cast<int>(value.rows);
    int lwork = 0;
    if (cusolverDnDpotrf_bufferSize(handle(), CUBLAS_FILL_MODE_UPPER, n, static_cast<double*>(factor.data()), n, &lwork) != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER potrf failed");
    }
    Device<double> buffer(static_cast<std::size_t>(std::max(lwork, 1)));
    Device<int> info(1);
    if (cusolverDnDpotrf(handle(), CUBLAS_FILL_MODE_UPPER, n, static_cast<double*>(factor.data()), n, buffer.data, lwork, info.data) != CUSOLVER_STATUS_SUCCESS ||
        readInfo(info.data) != 0) {
        throw std::runtime_error("matrix is not positive definite");
    }
    for (int c = 0; c < n; ++c) {
        for (int r = c + 1; r < n; ++r) factor.set(c * n + r, 0);
    }
    return finish(factor, type);
}

Value inverse(const Value& value) {
    if (value.rows != value.cols) throw std::runtime_error("inv requires a square matrix");
    const DType type = value.type == DType::F32 ? DType::F32 : DType::F64;
    LuWork lu = decompose(value);
    if (lu.matrix.rows == 0) return finish(lu.matrix, type);
    return finish(solve(lu, identity(value.rows)), type);
}

Value determinant(const Value& value) {
    if (value.rows != value.cols) throw std::runtime_error("det requires a square matrix");
    if (value.rows == 0) return Value::scalar(DType::F64, 1);
    LuWork lu = decompose(value);
    double product = 1;
    int sign = 1;
    for (std::int64_t i = 0; i < value.rows; ++i) {
        product *= lu.matrix.get(i * value.rows + i);
        if (i < static_cast<std::int64_t>(lu.pivot.size()) && lu.pivot[static_cast<std::size_t>(i)] != static_cast<int>(i + 1)) sign = -sign;
    }
    const DType type = value.type == DType::F32 ? DType::F32 : DType::F64;
    return Value::scalar(type, sign * product);
}

Value matrixRank(const Value& value) {
    const auto parts = factorSvd(value, 1);
    if (parts.empty() || parts[0].numel() == 0) return Value::scalar(DType::F64, 0);
    const double largest = std::abs(parts[0].get(0));
    const double tol = std::max(value.rows, value.cols) * largest * 2.220446049250313e-16;
    int count = 0;
    for (std::int64_t i = 0; i < parts[0].numel(); ++i) {
        if (std::abs(parts[0].get(i)) > tol) ++count;
    }
    return Value::scalar(DType::F64, count);
}

Value mldivide(const Value& left, const Value& right) {
    const DType type = left.type == DType::F32 && right.type == DType::F32 ? DType::F32 : DType::F64;
    if (left.rows != left.cols) throw std::runtime_error("backslash requires a square left matrix");
    LuWork lu = decompose(left);
    Value b = copyDouble(right);
    return finish(solve(lu, b), type);
}

Value mrdivide(const Value& left, const Value& right) {
    return transpose(mldivide(transpose(right), transpose(left)));
}

}  // namespace cudamath
