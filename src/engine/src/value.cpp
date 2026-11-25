#include "cudamath/value.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace cudamath {
namespace {

void requireCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::string trimNumber(double number) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.15g", number);
    return buffer;
}

}  // namespace

Buffer::~Buffer() {
    if (data != nullptr) {
        cudaFree(data);
        data = nullptr;
    }
}

const char* dtypeName(DType type) {
    switch (type) {
        case DType::F64: return "double";
        case DType::F32: return "single";
        case DType::I32: return "int32";
        case DType::I64: return "int64";
    }
    return "double";
}

std::size_t dtypeSize(DType type) {
    switch (type) {
        case DType::F64: return 8;
        case DType::F32: return 4;
        case DType::I32: return 4;
        case DType::I64: return 8;
    }
    return 8;
}

Value Value::str(std::string text) {
    Value value;
    value.isString = true;
    value.text = std::move(text);
    return value;
}

Value Value::matrix(DType type, std::int64_t rows, std::int64_t cols) {
    if (rows < 0 || cols < 0) {
        throw std::runtime_error("size must be nonnegative");
    }
    Value value;
    value.type = type;
    value.rows = rows;
    value.cols = cols;
    const std::size_t bytes = static_cast<std::size_t>(rows * cols) * dtypeSize(type);
    if (bytes == 0) {
        return value;
    }
    void* memory = nullptr;
    requireCuda(cudaMallocManaged(&memory, bytes, cudaMemAttachGlobal), "cudaMallocManaged");
    requireCuda(cudaMemset(memory, 0, bytes), "cudaMemset");
    value.buffer = std::shared_ptr<Buffer>(new Buffer{memory, bytes});
    return value;
}

Value Value::adopt(DType type, std::int64_t rows, std::int64_t cols, void* managed, std::size_t bytes) {
    Value value;
    value.type = type;
    value.rows = rows;
    value.cols = cols;
    value.buffer = std::shared_ptr<Buffer>(new Buffer{managed, bytes});
    return value;
}

Value Value::scalar(DType type, double number) {
    Value value = matrix(type, 1, 1);
    value.set(0, number);
    return value;
}

double Value::get(std::int64_t index) const {
    if (isString || data() == nullptr || index < 0 || index >= numel()) {
        throw std::runtime_error("index out of range");
    }
    const char* base = static_cast<const char*>(data()) + static_cast<std::size_t>(index) * dtypeSize(type);
    switch (type) {
        case DType::F64: return *reinterpret_cast<const double*>(base);
        case DType::F32: return *reinterpret_cast<const float*>(base);
        case DType::I32: return *reinterpret_cast<const std::int32_t*>(base);
        case DType::I64: return static_cast<double>(*reinterpret_cast<const std::int64_t*>(base));
    }
    return 0;
}

void Value::set(std::int64_t index, double number) {
    if (isString || data() == nullptr || index < 0 || index >= numel()) {
        throw std::runtime_error("index out of range");
    }
    char* base = static_cast<char*>(data()) + static_cast<std::size_t>(index) * dtypeSize(type);
    switch (type) {
        case DType::F64: *reinterpret_cast<double*>(base) = number; break;
        case DType::F32: *reinterpret_cast<float*>(base) = static_cast<float>(number); break;
        case DType::I32: *reinterpret_cast<std::int32_t*>(base) = static_cast<std::int32_t>(number); break;
        case DType::I64: *reinterpret_cast<std::int64_t*>(base) = static_cast<std::int64_t>(number); break;
    }
}

std::string Value::format() const {
    if (isString) {
        return text;
    }
    if (empty()) {
        return "[]";
    }
    if (numel() > 64) {
        return dtypeName(type) + std::string(" ") + std::to_string(rows) + "x" + std::to_string(cols);
    }
    std::ostringstream out;
    for (std::int64_t row = 0; row < rows; ++row) {
        if (row > 0) {
            out << '\n';
        }
        for (std::int64_t col = 0; col < cols; ++col) {
            if (col > 0) {
                out << ' ';
            }
            out << trimNumber(get(col * rows + row));
        }
    }
    return out.str();
}

std::string Value::whosLine(const std::string& name) const {
    if (isString) {
        return name + "  char  " + std::to_string(text.size());
    }
    return name + "  " + dtypeName(type) + "  " + std::to_string(rows) + "x" + std::to_string(cols);
}

bool isTrue(const Value& value) {
    if (value.isString) {
        return !value.text.empty();
    }
    if (value.empty()) {
        return false;
    }
    for (std::int64_t i = 0; i < value.numel(); ++i) {
        if (value.get(i) == 0) {
            return false;
        }
    }
    return true;
}

double asNumber(const Value& value) {
    if (value.isString || value.numel() != 1) {
        throw std::runtime_error("expected a scalar");
    }
    return value.get(0);
}

}  // namespace cudamath
