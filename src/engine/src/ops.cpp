#include "cudamath/ops.hpp"

#include "cudamath/engine.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cudamath {
namespace {

extern "C" void cm_add_f64(double*, const double*, const double*, std::size_t);
extern "C" void cm_sub_f64(double*, const double*, const double*, std::size_t);
extern "C" void cm_mul_f64(double*, const double*, const double*, std::size_t);
extern "C" void cm_div_f64(double*, const double*, const double*, std::size_t);
extern "C" void cm_add_f32(float*, const float*, const float*, std::size_t);
extern "C" void cm_sub_f32(float*, const float*, const float*, std::size_t);
extern "C" void cm_mul_f32(float*, const float*, const float*, std::size_t);
extern "C" void cm_div_f32(float*, const float*, const float*, std::size_t);
extern "C" void cm_add_i32(int*, const int*, const int*, std::size_t);
extern "C" void cm_sub_i32(int*, const int*, const int*, std::size_t);
extern "C" void cm_mul_i32(int*, const int*, const int*, std::size_t);
extern "C" void cm_add_i64(long long*, const long long*, const long long*, std::size_t);
extern "C" void cm_sub_i64(long long*, const long long*, const long long*, std::size_t);
extern "C" void cm_mul_i64(long long*, const long long*, const long long*, std::size_t);
extern "C" void cm_abs_f64(double*, const double*, std::size_t);
extern "C" void cm_sqrt_f64(double*, const double*, std::size_t);

void sync() {
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

Value filled(DType type, std::int64_t rows, std::int64_t cols, double number) {
    Value value = Value::matrix(type, rows, cols);
    for (std::int64_t i = 0; i < value.numel(); ++i) {
        value.set(i, number);
    }
    return value;
}

Value promote(const Value& value, DType type) {
    if (value.type == type) {
        return value;
    }
    Value out = Value::matrix(type, value.rows, value.cols);
    for (std::int64_t i = 0; i < value.numel(); ++i) {
        out.set(i, value.get(i));
    }
    return out;
}

Value scalarLike(const Value& shape, double number) {
    if (shape.numel() == 1) {
        return Value::scalar(shape.type, number);
    }
    return filled(shape.type, shape.rows, shape.cols, number);
}

void element(char op, DType type, void* c, const void* a, const void* b, std::size_t n) {
    switch (type) {
        case DType::F64:
            if (op == '+') cm_add_f64(static_cast<double*>(c), static_cast<const double*>(a), static_cast<const double*>(b), n);
            else if (op == '-') cm_sub_f64(static_cast<double*>(c), static_cast<const double*>(a), static_cast<const double*>(b), n);
            else if (op == '*') cm_mul_f64(static_cast<double*>(c), static_cast<const double*>(a), static_cast<const double*>(b), n);
            else cm_div_f64(static_cast<double*>(c), static_cast<const double*>(a), static_cast<const double*>(b), n);
            break;
        case DType::F32:
            if (op == '+') cm_add_f32(static_cast<float*>(c), static_cast<const float*>(a), static_cast<const float*>(b), n);
            else if (op == '-') cm_sub_f32(static_cast<float*>(c), static_cast<const float*>(a), static_cast<const float*>(b), n);
            else if (op == '*') cm_mul_f32(static_cast<float*>(c), static_cast<const float*>(a), static_cast<const float*>(b), n);
            else cm_div_f32(static_cast<float*>(c), static_cast<const float*>(a), static_cast<const float*>(b), n);
            break;
        case DType::I32:
            if (op == '+') cm_add_i32(static_cast<int*>(c), static_cast<const int*>(a), static_cast<const int*>(b), n);
            else if (op == '-') cm_sub_i32(static_cast<int*>(c), static_cast<const int*>(a), static_cast<const int*>(b), n);
            else cm_mul_i32(static_cast<int*>(c), static_cast<const int*>(a), static_cast<const int*>(b), n);
            break;
        case DType::I64:
            if (op == '+') cm_add_i64(static_cast<long long*>(c), static_cast<const long long*>(a), static_cast<const long long*>(b), n);
            else if (op == '-') cm_sub_i64(static_cast<long long*>(c), static_cast<const long long*>(a), static_cast<const long long*>(b), n);
            else cm_mul_i64(static_cast<long long*>(c), static_cast<const long long*>(a), static_cast<const long long*>(b), n);
            break;
    }
    sync();
}

void ensureRng(Session& session) {
    if (session.rngAlive) {
        return;
    }
    curandGenerator_t generator = nullptr;
    if (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10) != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("cuRAND could not create a generator");
    }
    curandSetPseudoRandomGeneratorSeed(generator, session.rngSeed);
    session.rng = generator;
    session.rngAlive = true;
    std::uint64_t left = session.rngDrawn;
    while (left > 0) {
        const std::size_t chunk = static_cast<std::size_t>(left > 1000000 ? 1000000 : left);
        Value skip = Value::matrix(DType::F64, static_cast<std::int64_t>(chunk), 1);
        curandGenerateUniformDouble(generator, static_cast<double*>(skip.data()), chunk);
        left -= chunk;
    }
}

Value randomDraw(Session& session, std::int64_t rows, std::int64_t cols, bool normal) {
    Value value = Value::matrix(DType::F64, rows, cols);
    ensureRng(session);
    auto generator = reinterpret_cast<curandGenerator_t>(session.rng);
    const std::size_t n = static_cast<std::size_t>(value.numel());
    const curandStatus_t status = normal
        ? curandGenerateNormalDouble(generator, static_cast<double*>(value.data()), n, 0.0, 1.0)
        : curandGenerateUniformDouble(generator, static_cast<double*>(value.data()), n);
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("cuRAND generation failed");
    }
    session.rngDrawn += n;
    sync();
    return value;
}

}  // namespace

Value zeros(DType type, std::int64_t rows, std::int64_t cols) { return Value::matrix(type, rows, cols); }

Value ones(DType type, std::int64_t rows, std::int64_t cols) { return filled(type, rows, cols, 1); }

Value eye(DType type, std::int64_t rows, std::int64_t cols) {
    Value value = Value::matrix(type, rows, cols);
    const std::int64_t n = rows < cols ? rows : cols;
    for (std::int64_t i = 0; i < n; ++i) {
        value.set(i * rows + i, 1);
    }
    return value;
}

Value reshape(const Value& value, std::int64_t rows, std::int64_t cols) {
    if (value.isString || rows * cols != value.numel()) {
        throw std::runtime_error("reshape element count must stay the same");
    }
    Value out = value;
    out.rows = rows;
    out.cols = cols;
    return out;
}

Value transpose(const Value& value) {
    if (value.isString) {
        throw std::runtime_error("cannot transpose a string");
    }
    Value out = Value::matrix(value.type, value.cols, value.rows);
    for (std::int64_t c = 0; c < value.cols; ++c) {
        for (std::int64_t r = 0; r < value.rows; ++r) {
            out.set(r * value.cols + c, value.get(c * value.rows + r));
        }
    }
    return out;
}

Value convertTo(const Value& value, DType type) {
    if (value.isString) {
        throw std::runtime_error("cannot convert a string");
    }
    return promote(value, type);
}

Value binary(const Value& left, const Value& right, char op, bool elementwise) {
    if (left.isString || right.isString) {
        throw std::runtime_error("arithmetic requires numeric arrays");
    }
    if (!elementwise && op == '*') {
        if (left.cols != right.rows) {
            throw std::runtime_error("inner dimensions must agree");
        }
        const Value a = (left.type == DType::F32 && right.type == DType::F32) ? left : promote(left, DType::F64);
        const Value b = (left.type == DType::F32 && right.type == DType::F32) ? right : promote(right, DType::F64);
        Value out = Value::matrix(a.type, a.rows, b.cols);
        auto handle = reinterpret_cast<cublasHandle_t>(Engine::instance().cublas());
        const double one = 1;
        const double zero = 0;
        const float oneS = 1;
        const float zeroS = 0;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
        if (a.type == DType::F32) {
            status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(a.rows), static_cast<int>(b.cols), static_cast<int>(a.cols),
                &oneS, static_cast<const float*>(a.data()), static_cast<int>(a.rows),
                static_cast<const float*>(b.data()), static_cast<int>(b.rows),
                &zeroS, static_cast<float*>(out.data()), static_cast<int>(out.rows));
        } else {
            status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(a.rows), static_cast<int>(b.cols), static_cast<int>(a.cols),
                &one, static_cast<const double*>(a.data()), static_cast<int>(a.rows),
                static_cast<const double*>(b.data()), static_cast<int>(b.rows),
                &zero, static_cast<double*>(out.data()), static_cast<int>(out.rows));
        }
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublas gemm failed");
        }
        sync();
        return out;
    }
    DType type = left.type;
    if (op == '/' || left.type != right.type || left.type == DType::F32 || right.type == DType::F32 || left.type == DType::F64 || right.type == DType::F64) {
        if (left.type == DType::F32 && right.type == DType::F32 && op != '/') {
            type = DType::F32;
        } else if (left.type == right.type && (left.type == DType::I32 || left.type == DType::I64) && op != '/') {
            type = left.type;
        } else {
            type = DType::F64;
        }
    }
    Value a = promote(left, type);
    Value b = promote(right, type);
    if (a.numel() == 1 && b.numel() != 1) a = scalarLike(b, a.get(0));
    if (b.numel() == 1 && a.numel() != 1) b = scalarLike(a, b.get(0));
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::runtime_error("array sizes must match");
    }
    Value out = Value::matrix(type, a.rows, a.cols);
    element(op, type, out.data(), a.data(), b.data(), static_cast<std::size_t>(out.numel()));
    return out;
}

Value compare(const Value& left, const Value& right, const char* op) {
    Value out = Value::matrix(DType::F64, left.rows, left.cols);
    if (left.numel() != right.numel() && left.numel() != 1 && right.numel() != 1) {
        throw std::runtime_error("array sizes must match");
    }
    if (left.numel() == right.numel()) {
        out = Value::matrix(DType::F64, left.rows, left.cols);
    } else if (left.numel() == 1) {
        out = Value::matrix(DType::F64, right.rows, right.cols);
    }
    for (std::int64_t i = 0; i < out.numel(); ++i) {
        const double a = left.numel() == 1 ? left.get(0) : left.get(i);
        const double b = right.numel() == 1 ? right.get(0) : right.get(i);
        double result = 0;
        if (op[0] == '=' ) result = a == b;
        else if (op[0] == '~') result = a != b;
        else if (op[0] == '<' && op[1] == '=') result = a <= b;
        else if (op[0] == '>' && op[1] == '=') result = a >= b;
        else if (op[0] == '<') result = a < b;
        else result = a > b;
        out.set(i, result);
    }
    return out;
}

Value unaryMinus(const Value& value) {
    return binary(Value::scalar(value.type, 0), value, '-', true);
}

Value applyAbs(const Value& value) {
    Value src = value.type == DType::F64 ? value : promote(value, DType::F64);
    Value out = Value::matrix(DType::F64, src.rows, src.cols);
    cm_abs_f64(static_cast<double*>(out.data()), static_cast<const double*>(src.data()), static_cast<std::size_t>(src.numel()));
    sync();
    return out;
}

Value applySqrt(const Value& value) {
    Value src = promote(value, DType::F64);
    Value out = Value::matrix(DType::F64, src.rows, src.cols);
    cm_sqrt_f64(static_cast<double*>(out.data()), static_cast<const double*>(src.data()), static_cast<std::size_t>(src.numel()));
    sync();
    return out;
}

Value sumAll(const Value& value) {
    double total = 0;
    for (std::int64_t i = 0; i < value.numel(); ++i) {
        total += value.get(i);
    }
    return Value::scalar(value.type == DType::I32 || value.type == DType::I64 ? value.type : DType::F64, total);
}

Value randUniform(Session& session, std::int64_t rows, std::int64_t cols, DType type) {
    Value drawn = randomDraw(session, rows, cols, false);
    return type == DType::F64 ? drawn : promote(drawn, type);
}

Value randNormal(Session& session, std::int64_t rows, std::int64_t cols, DType type) {
    Value drawn = randomDraw(session, rows, cols, true);
    return type == DType::F64 ? drawn : promote(drawn, type);
}

void seedRng(Session& session, std::uint64_t seed) {
    session.resetRng();
    session.rngSeed = seed;
    session.rngDrawn = 0;
}

}  // namespace cudamath
