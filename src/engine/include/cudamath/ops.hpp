#pragma once

#include "cudamath/session.hpp"

namespace cudamath {

Value zeros(DType type, std::int64_t rows, std::int64_t cols);
Value ones(DType type, std::int64_t rows, std::int64_t cols);
Value eye(DType type, std::int64_t rows, std::int64_t cols);
Value reshape(const Value& value, std::int64_t rows, std::int64_t cols);
Value transpose(const Value& value);
Value convertTo(const Value& value, DType type);
Value binary(const Value& left, const Value& right, char op, bool elementwise);
Value compare(const Value& left, const Value& right, const char* op);
Value unaryMinus(const Value& value);
Value applyAbs(const Value& value);
Value applySqrt(const Value& value);
Value sumAll(const Value& value);
Value randUniform(Session& session, std::int64_t rows, std::int64_t cols, DType type);
Value randNormal(Session& session, std::int64_t rows, std::int64_t cols, DType type);
void seedRng(Session& session, std::uint64_t seed);

}  // namespace cudamath
