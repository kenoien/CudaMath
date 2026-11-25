#include "cudamath/ops.hpp"

#include "cudamath/engine.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cudamath {

Value cloneArray(const Value& value);

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
extern "C" void cm_unary_f64(int, double*, const double*, std::size_t);
extern "C" void cm_unary_f32(int, float*, const float*, std::size_t);
extern "C" void cm_unary_i32(int, int*, const int*, std::size_t);
extern "C" void cm_unary_i64(int, long long*, const long long*, std::size_t);
extern "C" void cm_binary_f64(int, double*, const double*, const double*, std::size_t);
extern "C" void cm_binary_f32(int, float*, const float*, const float*, std::size_t);
extern "C" void cm_col_f64(int, double*, const double*, int, int);
extern "C" void cm_col_f32(int, float*, const float*, int, int);
extern "C" void cm_col_i32(int, int*, const int*, int, int);
extern "C" void cm_col_i64(int, long long*, const long long*, int, int);
extern "C" void cm_row_f64(int, double*, const double*, int, int);
extern "C" void cm_row_f32(int, float*, const float*, int, int);
extern "C" void cm_row_i32(int, int*, const int*, int, int);
extern "C" void cm_row_i64(int, long long*, const long long*, int, int);
extern "C" void cm_cross_f64(double*, const double*, const double*);

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
        const bool leftScalar = left.numel() == 1;
        const bool rightScalar = right.numel() == 1;
        if (leftScalar != rightScalar) {
            const Value& factor = leftScalar ? left : right;
            const Value& matrix = leftScalar ? right : left;
            const bool single = factor.type == DType::F32 && matrix.type == DType::F32;
            Value out = cloneArray(promote(matrix, single ? DType::F32 : DType::F64));
            if (out.numel() > 0) {
                const Value alphaValue = promote(factor, out.type);
                auto handle = reinterpret_cast<cublasHandle_t>(Engine::instance().cublas());
                cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
                if (out.type == DType::F32) {
                    const float alpha = static_cast<float>(alphaValue.get(0));
                    status = cublasSscal(handle, static_cast<int>(out.numel()), &alpha, static_cast<float*>(out.data()), 1);
                } else {
                    const double alpha = alphaValue.get(0);
                    status = cublasDscal(handle, static_cast<int>(out.numel()), &alpha, static_cast<double*>(out.data()), 1);
                }
                if (status != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublas scal failed");
                }
                sync();
            }
            return out;
        }
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
    if (value.type == DType::I32 || value.type == DType::I64) return applyUnary(value, 0);
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

Value cloneArray(const Value& value) {
    Value out = Value::matrix(value.type, value.rows, value.cols);
    if (value.numel() > 0 && value.data() != nullptr) {
        const cudaError_t status = cudaMemcpy(out.data(), value.data(), static_cast<std::size_t>(value.numel()) * dtypeSize(value.type), cudaMemcpyDefault);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        sync();
    }
    return out;
}

Value broadcastPair(const Value& left, const Value& right, Value& a, Value& b) {
    DType type = DType::F64;
    if (left.type == DType::F32 && right.type == DType::F32) type = DType::F32;
    a = promote(left, type);
    b = promote(right, type);
    if (a.numel() == 1 && b.numel() != 1) a = scalarLike(b, a.get(0));
    if (b.numel() == 1 && a.numel() != 1) b = scalarLike(a, b.get(0));
    if (a.rows != b.rows || a.cols != b.cols) throw std::runtime_error("array sizes must match");
    return Value::matrix(type, a.rows, a.cols);
}

Value applyUnary(const Value& value, int op) {
    if (value.isString) throw std::runtime_error("math requires a numeric array");
    const bool integer = value.type == DType::I32 || value.type == DType::I64;
    if (integer && op > 1) {
        Value src = promote(value, DType::F64);
        Value out = Value::matrix(DType::F64, src.rows, src.cols);
        cm_unary_f64(op, static_cast<double*>(out.data()), static_cast<const double*>(src.data()), static_cast<std::size_t>(src.numel()));
        sync();
        return out;
    }
    Value out = cloneArray(value.type == DType::F32 || integer || value.type == DType::F64 ? value : promote(value, DType::F64));
    const auto n = static_cast<std::size_t>(out.numel());
    if (out.type == DType::F64) cm_unary_f64(op, static_cast<double*>(out.data()), static_cast<const double*>(out.data()), n);
    else if (out.type == DType::F32) cm_unary_f32(op, static_cast<float*>(out.data()), static_cast<const float*>(out.data()), n);
    else if (out.type == DType::I32) cm_unary_i32(op, static_cast<int*>(out.data()), static_cast<const int*>(out.data()), n);
    else cm_unary_i64(op, static_cast<long long*>(out.data()), static_cast<const long long*>(out.data()), n);
    sync();
    return out;
}

Value applySign(const Value& value) { return applyUnary(value, 1); }

Value applyBinaryMath(const Value& left, const Value& right, int op) {
    const Value aSrc = left.type == DType::F32 || left.type == DType::F64 ? left : promote(left, DType::F64);
    const Value bSrc = right.type == DType::F32 || right.type == DType::F64 ? right : promote(right, DType::F64);
    Value a;
    Value b;
    Value out = broadcastPair(aSrc, bSrc, a, b);
    const auto n = static_cast<std::size_t>(out.numel());
    if (out.type == DType::F32) {
        cm_binary_f32(op, static_cast<float*>(out.data()), static_cast<const float*>(a.data()), static_cast<const float*>(b.data()), n);
    } else {
        cm_binary_f64(op, static_cast<double*>(out.data()), static_cast<const double*>(a.data()), static_cast<const double*>(b.data()), n);
    }
    sync();
    return out;
}

Value reduce(const Value& value, int op, int dim, bool all) {
    if (value.isString) throw std::runtime_error("reduction requires a numeric array");
    if (all) {
        if (op == 0) return sumAll(value);
        Value flat = reduce(value, op, 1, false);
        return reduce(flat, op, 2, false);
    }
    const bool down = dim != 2;
    const std::int64_t rows = down ? 1 : value.rows;
    const std::int64_t cols = down ? value.cols : 1;
    DType type = value.type;
    if (op == 0 || op == 1) {
        if (type != DType::I32 && type != DType::I64) type = DType::F64;
    } else if (op >= 4) {
        type = DType::F64;
    }
    Value src = value.type == type ? value : promote(value, type);
    Value out = Value::matrix(type, rows, cols);
    const int r = static_cast<int>(value.rows);
    const int c = static_cast<int>(value.cols);
    if (type == DType::F64) {
        auto* dest = static_cast<double*>(out.data());
        auto* source = static_cast<const double*>(src.data());
        if (down) cm_col_f64(op, dest, source, r, c);
        else cm_row_f64(op, dest, source, r, c);
    } else if (type == DType::F32) {
        auto* dest = static_cast<float*>(out.data());
        auto* source = static_cast<const float*>(src.data());
        if (down) cm_col_f32(op, dest, source, r, c);
        else cm_row_f32(op, dest, source, r, c);
    } else if (type == DType::I32) {
        auto* dest = static_cast<int*>(out.data());
        auto* source = static_cast<const int*>(src.data());
        if (down) cm_col_i32(op, dest, source, r, c);
        else cm_row_i32(op, dest, source, r, c);
    } else {
        auto* dest = static_cast<long long*>(out.data());
        auto* source = static_cast<const long long*>(src.data());
        if (down) cm_col_i64(op, dest, source, r, c);
        else cm_row_i64(op, dest, source, r, c);
    }
    sync();
    return out;
}

Value elementMinMax(const Value& left, const Value& right, bool isMin) {
    if (left.isString || right.isString) throw std::runtime_error("min and max require numeric arrays");
    if (left.numel() != right.numel() && left.numel() != 1 && right.numel() != 1) {
        throw std::runtime_error("array sizes must match");
    }
    const Value& shape = left.numel() == 1 ? right : left;
    const bool sameInteger = (left.type == DType::I32 || left.type == DType::I64) && left.type == right.type;
    const DType type = sameInteger ? left.type : (left.type == DType::F32 && right.type == DType::F32 ? DType::F32 : DType::F64);
    Value out = Value::matrix(type, shape.rows, shape.cols);
    for (std::int64_t i = 0; i < out.numel(); ++i) {
        const double a = left.numel() == 1 ? left.get(0) : left.get(i);
        const double b = right.numel() == 1 ? right.get(0) : right.get(i);
        out.set(i, isMin ? (std::min)(a, b) : (std::max)(a, b));
    }
    return out;
}

Value minimum(const Value& left, const Value& right) { return elementMinMax(left, right, true); }
Value maximum(const Value& left, const Value& right) { return elementMinMax(left, right, false); }

Value placed(DType type, std::int64_t rows, std::int64_t cols, const std::vector<double>& numbers) {
    Value out = Value::matrix(type, rows, cols);
    for (std::int64_t i = 0; i < out.numel() && i < static_cast<std::int64_t>(numbers.size()); ++i) out.set(i, numbers[static_cast<std::size_t>(i)]);
    return out;
}

Value diag(const Value& value) {
    if (value.isString) throw std::runtime_error("diag requires a numeric array");
    if (value.rows == 1 || value.cols == 1) {
        const std::int64_t n = value.numel();
        Value out = Value::matrix(value.type, n, n);
        for (std::int64_t i = 0; i < n; ++i) out.set(i * n + i, value.get(i));
        return out;
    }
    const std::int64_t n = std::min(value.rows, value.cols);
    Value out = Value::matrix(value.type, n, 1);
    for (std::int64_t i = 0; i < n; ++i) out.set(i, value.get(i * value.rows + i));
    return out;
}

Value linspace(double start, double stop, std::int64_t count) {
    if (count < 1) throw std::runtime_error("linspace count must be positive");
    Value out = Value::matrix(DType::F64, 1, count);
    if (count == 1) {
        out.set(0, stop);
        return out;
    }
    for (std::int64_t i = 0; i < count; ++i) {
        out.set(i, start + (stop - start) * static_cast<double>(i) / static_cast<double>(count - 1));
    }
    return out;
}

Value repmat(const Value& value, std::int64_t rowTiles, std::int64_t colTiles) {
    if (rowTiles < 0 || colTiles < 0) throw std::runtime_error("repmat size must be nonnegative");
    Value out = Value::matrix(value.type, value.rows * rowTiles, value.cols * colTiles);
    for (std::int64_t tr = 0; tr < rowTiles; ++tr) {
        for (std::int64_t tc = 0; tc < colTiles; ++tc) {
            for (std::int64_t c = 0; c < value.cols; ++c) {
                for (std::int64_t r = 0; r < value.rows; ++r) {
                    const std::int64_t row = tr * value.rows + r;
                    const std::int64_t col = tc * value.cols + c;
                    out.set(col * out.rows + row, value.get(c * value.rows + r));
                }
            }
        }
    }
    return out;
}

Value triangle(const Value& value, std::int64_t k, bool lower) {
    Value out = cloneArray(value);
    for (std::int64_t c = 0; c < value.cols; ++c) {
        for (std::int64_t r = 0; r < value.rows; ++r) {
            const bool keep = lower ? r >= c - k : r <= c - k;
            if (!keep) out.set(c * value.rows + r, 0);
        }
    }
    return out;
}

Value tril(const Value& value, std::int64_t k) { return triangle(value, k, true); }
Value triu(const Value& value, std::int64_t k) { return triangle(value, k, false); }

DType commonType(const std::vector<Value>& parts) {
    DType type = parts.front().type;
    for (const Value& part : parts) {
        if (part.type != type) return DType::F64;
    }
    return type;
}

Value horzcat(const std::vector<Value>& parts) {
    if (parts.empty()) return Value::matrix(DType::F64, 0, 0);
    const std::int64_t rows = parts.front().rows;
    std::int64_t cols = 0;
    for (const Value& part : parts) {
        if (part.rows != rows) throw std::runtime_error("horzcat rows must agree");
        cols += part.cols;
    }
    const DType type = commonType(parts);
    Value out = Value::matrix(type, rows, cols);
    std::int64_t col = 0;
    for (const Value& part : parts) {
        const Value src = part.type == type ? part : promote(part, type);
        for (std::int64_t c = 0; c < part.cols; ++c) {
            for (std::int64_t r = 0; r < rows; ++r) out.set((col + c) * rows + r, src.get(c * part.rows + r));
        }
        col += part.cols;
    }
    return out;
}

Value vertcat(const std::vector<Value>& parts) {
    if (parts.empty()) return Value::matrix(DType::F64, 0, 0);
    const std::int64_t cols = parts.front().cols;
    std::int64_t rows = 0;
    for (const Value& part : parts) {
        if (part.cols != cols) throw std::runtime_error("vertcat columns must agree");
        rows += part.rows;
    }
    const DType type = commonType(parts);
    Value out = Value::matrix(type, rows, cols);
    std::int64_t row = 0;
    for (const Value& part : parts) {
        const Value src = part.type == type ? part : promote(part, type);
        for (std::int64_t c = 0; c < cols; ++c) {
            for (std::int64_t r = 0; r < part.rows; ++r) out.set(c * rows + row + r, src.get(c * part.rows + r));
        }
        row += part.rows;
    }
    return out;
}

Value logicalNot(const Value& value) {
    Value out = Value::matrix(DType::F64, value.rows, value.cols);
    for (std::int64_t i = 0; i < value.numel(); ++i) out.set(i, value.get(i) == 0 ? 1 : 0);
    return out;
}

Value logicalCombine(const Value& left, const Value& right, bool conjunction) {
    if (left.numel() != right.numel() && left.numel() != 1 && right.numel() != 1) throw std::runtime_error("array sizes must match");
    const std::int64_t rows = left.numel() == 1 ? right.rows : left.rows;
    const std::int64_t cols = left.numel() == 1 ? right.cols : left.cols;
    Value out = Value::matrix(DType::F64, rows, cols);
    for (std::int64_t i = 0; i < out.numel(); ++i) {
        const bool a = (left.numel() == 1 ? left.get(0) : left.get(i)) != 0;
        const bool b = (right.numel() == 1 ? right.get(0) : right.get(i)) != 0;
        out.set(i, conjunction ? (a && b) : (a || b));
    }
    return out;
}

Value logicalAnd(const Value& left, const Value& right) { return logicalCombine(left, right, true); }
Value logicalOr(const Value& left, const Value& right) { return logicalCombine(left, right, false); }

Value subscript(const Value& value, const std::vector<Value>& indexes, const std::vector<bool>& colon) {
    auto positions = [&](std::size_t axis, std::int64_t length) {
        std::vector<std::int64_t> chosen;
        if (colon[axis]) {
            chosen.resize(static_cast<std::size_t>(length));
            for (std::int64_t i = 0; i < length; ++i) chosen[static_cast<std::size_t>(i)] = i;
            return chosen;
        }
        const Value& index = indexes[axis];
        for (std::int64_t i = 0; i < index.numel(); ++i) {
            const auto pos = static_cast<std::int64_t>(index.get(i));
            if (pos < 1 || pos > length) throw std::runtime_error("index out of range");
            chosen.push_back(pos - 1);
        }
        return chosen;
    };
    if (colon.size() == 1 && colon[0]) {
        Value out = Value::matrix(value.type, value.numel(), 1);
        for (std::int64_t i = 0; i < value.numel(); ++i) out.set(i, value.get(i));
        return out;
    }
    if (colon.size() == 1 && !colon[0]) {
        const auto chosen = positions(0, value.numel());
        Value out = Value::matrix(value.type, static_cast<std::int64_t>(chosen.size()), 1);
        if (value.rows == 1) out = Value::matrix(value.type, 1, static_cast<std::int64_t>(chosen.size()));
        for (std::size_t i = 0; i < chosen.size(); ++i) out.set(static_cast<std::int64_t>(i), value.get(chosen[i]));
        return out;
    }
    const auto rowIndex = positions(0, value.rows);
    const auto colIndex = colon.size() < 2 ? positions(0, value.cols) : positions(1, value.cols);
    if (colon.size() < 2) {
        Value out = Value::matrix(value.type, 1, static_cast<std::int64_t>(colIndex.size()));
        for (std::size_t c = 0; c < colIndex.size(); ++c) out.set(static_cast<std::int64_t>(c), value.get(colIndex[c] * value.rows));
        return out;
    }
    Value out = Value::matrix(value.type, static_cast<std::int64_t>(rowIndex.size()), static_cast<std::int64_t>(colIndex.size()));
    for (std::size_t c = 0; c < colIndex.size(); ++c) {
        for (std::size_t r = 0; r < rowIndex.size(); ++r) {
            out.set(static_cast<std::int64_t>(c) * out.rows + static_cast<std::int64_t>(r), value.get(colIndex[c] * value.rows + rowIndex[r]));
        }
    }
    return out;
}

std::vector<double> vectorNumbers(const Value& value) {
    if (!(value.rows == 1 || value.cols == 1)) throw std::runtime_error("expected a vector");
    std::vector<double> numbers(static_cast<std::size_t>(value.numel()));
    for (std::int64_t i = 0; i < value.numel(); ++i) numbers[static_cast<std::size_t>(i)] = value.get(i);
    return numbers;
}

Value dot(const Value& left, const Value& right) {
    const auto a = vectorNumbers(left);
    const auto b = vectorNumbers(right);
    if (a.size() != b.size()) throw std::runtime_error("dot vectors must have the same length");
    Value x = promote(left.rows == 1 || left.cols == 1 ? left : left, left.type == DType::F32 && right.type == DType::F32 ? DType::F32 : DType::F64);
    Value y = promote(right, x.type);
    auto handle = reinterpret_cast<cublasHandle_t>(Engine::instance().cublas());
    if (x.type == DType::F32) {
        float result = 0;
        if (cublasSdot(handle, static_cast<int>(a.size()), static_cast<const float*>(x.data()), 1, static_cast<const float*>(y.data()), 1, &result) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublas dot failed");
        }
        sync();
        return Value::scalar(DType::F32, result);
    }
    double result = 0;
    x = promote(left, DType::F64);
    y = promote(right, DType::F64);
    if (cublasDdot(handle, static_cast<int>(a.size()), static_cast<const double*>(x.data()), 1, static_cast<const double*>(y.data()), 1, &result) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublas dot failed");
    }
    sync();
    return Value::scalar(DType::F64, result);
}

Value cross(const Value& left, const Value& right) {
    const auto a = vectorNumbers(promote(left, DType::F64));
    const auto b = vectorNumbers(promote(right, DType::F64));
    if (a.size() != 3 || b.size() != 3) throw std::runtime_error("cross requires 3-vectors");
    Value x = Value::matrix(DType::F64, 3, 1);
    Value y = Value::matrix(DType::F64, 3, 1);
    for (int i = 0; i < 3; ++i) {
        x.set(i, a[static_cast<std::size_t>(i)]);
        y.set(i, b[static_cast<std::size_t>(i)]);
    }
    const bool row = left.rows == 1 && right.rows == 1;
    Value out = Value::matrix(DType::F64, row ? 1 : 3, row ? 3 : 1);
    Value tmp = Value::matrix(DType::F64, 3, 1);
    cm_cross_f64(static_cast<double*>(tmp.data()), static_cast<const double*>(x.data()), static_cast<const double*>(y.data()));
    sync();
    for (int i = 0; i < 3; ++i) out.set(i, tmp.get(i));
    return out;
}

Value trace(const Value& value) {
    double total = 0;
    const std::int64_t n = std::min(value.rows, value.cols);
    for (std::int64_t i = 0; i < n; ++i) total += value.get(i * value.rows + i);
    return Value::scalar(value.type == DType::I32 || value.type == DType::I64 ? value.type : DType::F64, total);
}

Value norm(const Value& value, int kind) {
    const bool vector = value.rows == 1 || value.cols == 1;
    if (kind == 0 || (kind == 2 && vector) || (kind < 0 && vector)) {
        if (kind == 0 || kind == 2 || kind < 0) {
            if (kind != 0 && vector) {
                Value src = promote(value, value.type == DType::F32 ? DType::F32 : DType::F64);
                auto handle = reinterpret_cast<cublasHandle_t>(Engine::instance().cublas());
                if (src.type == DType::F32 && kind != 0) {
                    float result = 0;
                    if (cublasSnrm2(handle, static_cast<int>(src.numel()), static_cast<const float*>(src.data()), 1, &result) != CUBLAS_STATUS_SUCCESS) {
                        throw std::runtime_error("cublas nrm2 failed");
                    }
                    sync();
                    return Value::scalar(DType::F32, result);
                }
                src = promote(value, DType::F64);
                double result = 0;
                if (cublasDnrm2(handle, static_cast<int>(src.numel()), static_cast<const double*>(src.data()), 1, &result) != CUBLAS_STATUS_SUCCESS) {
                    throw std::runtime_error("cublas nrm2 failed");
                }
                sync();
                if (kind == 0) {
                    double total = 0;
                    for (std::int64_t i = 0; i < value.numel(); ++i) total += value.get(i) * value.get(i);
                    return Value::scalar(DType::F64, std::sqrt(total));
                }
                return Value::scalar(DType::F64, result);
            }
        }
    }
    if (kind == 0) {
        double total = 0;
        for (std::int64_t i = 0; i < value.numel(); ++i) total += value.get(i) * value.get(i);
        return Value::scalar(DType::F64, std::sqrt(total));
    }
    if (kind == 1) {
        if (vector) {
            double total = 0;
            for (std::int64_t i = 0; i < value.numel(); ++i) total += std::abs(value.get(i));
            return Value::scalar(DType::F64, total);
        }
        double best = 0;
        for (std::int64_t c = 0; c < value.cols; ++c) {
            double total = 0;
            for (std::int64_t r = 0; r < value.rows; ++r) total += std::abs(value.get(c * value.rows + r));
            best = std::max(best, total);
        }
        return Value::scalar(DType::F64, best);
    }
    if (kind == 3) {
        double best = 0;
        if (vector) {
            for (std::int64_t i = 0; i < value.numel(); ++i) best = std::max(best, std::abs(value.get(i)));
            return Value::scalar(DType::F64, best);
        }
        for (std::int64_t r = 0; r < value.rows; ++r) {
            double total = 0;
            for (std::int64_t c = 0; c < value.cols; ++c) total += std::abs(value.get(c * value.rows + r));
            best = std::max(best, total);
        }
        return Value::scalar(DType::F64, best);
    }
    const auto parts = factorSvd(value, 1);
    return Value::scalar(DType::F64, parts.empty() || parts[0].numel() == 0 ? 0 : parts[0].get(0));
}

}  // namespace cudamath
