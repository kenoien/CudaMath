#pragma once

#include "cudamath/session.hpp"

#include <vector>

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

Value applyUnary(const Value& value, int op);
Value applySign(const Value& value);
Value applyBinaryMath(const Value& left, const Value& right, int op);
Value reduce(const Value& value, int op, int dim, bool all);
Value minimum(const Value& left, const Value& right);
Value maximum(const Value& left, const Value& right);
Value diag(const Value& value);
Value linspace(double start, double stop, std::int64_t count);
Value repmat(const Value& value, std::int64_t rows, std::int64_t cols);
Value tril(const Value& value, std::int64_t k);
Value triu(const Value& value, std::int64_t k);
Value horzcat(const std::vector<Value>& parts);
Value vertcat(const std::vector<Value>& parts);
Value logicalNot(const Value& value);
Value logicalAnd(const Value& left, const Value& right);
Value logicalOr(const Value& left, const Value& right);
Value subscript(const Value& value, const std::vector<Value>& indexes, const std::vector<bool>& colon);
Value dot(const Value& left, const Value& right);
Value cross(const Value& left, const Value& right);
Value trace(const Value& value);
Value norm(const Value& value, int kind);
std::vector<Value> factorLu(const Value& value, int outputs);
std::vector<Value> factorQr(const Value& value, int outputs);
std::vector<Value> factorSvd(const Value& value, int outputs);
std::vector<Value> factorEig(const Value& value, int outputs);
Value chol(const Value& value);
Value inverse(const Value& value);
Value determinant(const Value& value);
Value matrixRank(const Value& value);
Value mldivide(const Value& left, const Value& right);
Value mrdivide(const Value& left, const Value& right);

}  // namespace cudamath
