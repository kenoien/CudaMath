#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace cudamath {

enum class DType { F64, F32, I32, I64 };

const char* dtypeName(DType type);
std::size_t dtypeSize(DType type);

struct Buffer {
    void* data{nullptr};
    std::size_t bytes{0};
    ~Buffer();
};

// A dense column-major array in unified memory, or a host string used for
// cudaversion and file paths. Arrays are shared by shared_ptr so assignment
// and ans keep the allocation alive.
struct Value {
    bool isString{false};
    std::string text;
    DType type{DType::F64};
    std::int64_t rows{0};
    std::int64_t cols{0};
    std::shared_ptr<Buffer> buffer;

    bool empty() const { return !isString && (rows == 0 || cols == 0); }
    std::int64_t numel() const { return isString ? 0 : rows * cols; }
    void* data() const { return buffer ? buffer->data : nullptr; }

    static Value str(std::string text);
    static Value scalar(DType type, double number);
    static Value matrix(DType type, std::int64_t rows, std::int64_t cols);
    static Value adopt(DType type, std::int64_t rows, std::int64_t cols, void* managed, std::size_t bytes);

    double get(std::int64_t index) const;
    void set(std::int64_t index, double number);
    std::string format() const;
    std::string whosLine(const std::string& name) const;
};

bool isTrue(const Value& value);
double asNumber(const Value& value);

}  // namespace cudamath
