#include "cudamath/ops.hpp"
#include "cudamath/repl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cudamath {
namespace {

struct Frame {
    std::map<std::string, Value>* locals{nullptr};
};

class EvalError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Parser {
public:
    Parser(Session& session, std::string source, Frame frame)
        : session_(session), source_(std::move(source)), frame_(frame) {}

    std::string run() {
        std::ostringstream out;
        while (!done()) {
            skip();
            if (done()) break;
            std::string piece = statement(out);
            if (piece == "\x01" || piece == "\x02") return piece;
            const bool suppress = eat(';');
            if (!piece.empty() && !suppress) {
                out << piece;
                if (piece.back() != '\n') out << '\n';
            }
        }
        return out.str();
    }

private:
    Session& session_;
    std::string source_;
    std::size_t i_{0};
    Frame frame_;
    int nargout_{1};
    int indexDepth_{0};
    std::int64_t indexEnd_{0};
    std::vector<Value> multi_;

    bool done() const { return i_ >= source_.size(); }
    char peek() const { return done() ? '\0' : source_[i_]; }

    void skip() {
        while (!done()) {
            if (std::isspace(static_cast<unsigned char>(peek()))) {
                ++i_;
                continue;
            }
            if (peek() == '%') {
                while (!done() && peek() != '\n') ++i_;
                continue;
            }
            break;
        }
    }

    bool eat(char c) {
        skip();
        if (peek() != c) return false;
        ++i_;
        return true;
    }

    bool eatWord(const char* word) {
        skip();
        const std::size_t mark = i_;
        for (const char* p = word; *p; ++p) {
            char c = peek();
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != *p) {
                i_ = mark;
                return false;
            }
            ++i_;
        }
        const char after = peek();
        if ((after >= 'a' && after <= 'z') || (after >= 'A' && after <= 'Z') || (after >= '0' && after <= '9') || after == '_') {
            i_ = mark;
            return false;
        }
        return true;
    }

    std::string ident() {
        skip();
        if (!std::isalpha(static_cast<unsigned char>(peek())) && peek() != '_') {
            throw EvalError("expected a name");
        }
        std::string name;
        while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_') {
            name.push_back(peek());
            ++i_;
        }
        return name;
    }

    std::string quoted() {
        skip();
        if (!eat('\'')) throw EvalError("expected a string");
        std::string text;
        while (!done() && peek() != '\'') {
            text.push_back(peek());
            ++i_;
        }
        if (!eat('\'')) throw EvalError("unterminated string");
        return text;
    }

    Value* lookup(const std::string& name) {
        if (frame_.locals != nullptr) {
            auto it = frame_.locals->find(name);
            if (it != frame_.locals->end()) return &it->second;
        }
        auto it = session_.vars.find(name);
        if (it == session_.vars.end()) return nullptr;
        return &it->second;
    }

    void assign(const std::string& name, Value value) {
        if (frame_.locals != nullptr) (*frame_.locals)[name] = std::move(value);
        else {
            session_.vars[name] = std::move(value);
            session_.dirty = true;
        }
    }

    std::int64_t positiveSize(const Value& value) {
        const double number = asNumber(value);
        if (number < 0) throw EvalError("size must be nonnegative");
        return static_cast<std::int64_t>(number);
    }

    DType typeArg(const std::vector<Value>& args, std::size_t index, DType fallback) {
        if (args.size() <= index || !args[index].isString) return fallback;
        const std::string& name = args[index].text;
        if (name == "double") return DType::F64;
        if (name == "single") return DType::F32;
        if (name == "int32") return DType::I32;
        if (name == "int64") return DType::I64;
        throw EvalError("unknown type " + name);
    }

    Value call(const std::string& name, const std::vector<Value>& args) {
        if (name == "zeros" || name == "ones" || name == "eye" || name == "rand" || name == "randn") {
            if (args.empty()) throw EvalError(name + " needs a size");
            const std::int64_t rows = positiveSize(args[0]);
            const std::int64_t cols = args.size() >= 2 && !args[1].isString ? positiveSize(args[1]) : rows;
            const std::size_t typeIndex = args.size() >= 2 && args[1].isString ? 1 : (args.size() >= 3 ? 2 : 99);
            const DType type = typeArg(args, typeIndex, DType::F64);
            if (name == "zeros") return zeros(type, rows, cols);
            if (name == "ones") return ones(type, rows, cols);
            if (name == "eye") return eye(type, rows, cols);
            if (name == "rand") return randUniform(session_, rows, cols, type == DType::I32 || type == DType::I64 ? DType::F64 : type);
            return randNormal(session_, rows, cols, type == DType::I32 || type == DType::I64 ? DType::F64 : type);
        }
        if (name == "double" || name == "single" || name == "int32" || name == "int64") {
            if (args.size() != 1) throw EvalError(name + " needs one argument");
            DType type = DType::F64;
            if (name == "single") type = DType::F32;
            if (name == "int32") type = DType::I32;
            if (name == "int64") type = DType::I64;
            return convertTo(args[0], type);
        }
        if (name == "size") {
            if (args.empty()) throw EvalError("size needs an array");
            if (args.size() == 1) {
                Value out = Value::matrix(DType::F64, 1, 2);
                out.set(0, static_cast<double>(args[0].rows));
                out.set(1, static_cast<double>(args[0].cols));
                return out;
            }
            const int dim = static_cast<int>(asNumber(args[1]));
            return Value::scalar(DType::F64, dim == 2 ? static_cast<double>(args[0].cols) : static_cast<double>(args[0].rows));
        }
        if (name == "numel") return Value::scalar(DType::F64, static_cast<double>(args.at(0).numel()));
        if (name == "length") {
            const Value& a = args.at(0);
            return Value::scalar(DType::F64, static_cast<double>(a.rows > a.cols ? a.rows : a.cols));
        }
        if (name == "ndims") return Value::scalar(DType::F64, 2);
        if (name == "isempty") return Value::scalar(DType::F64, args.at(0).empty() ? 1 : 0);
        if (name == "class") return Value::str(args.at(0).isString ? "char" : dtypeName(args[0].type));
        if (name == "print" || name == "prn") {
            if (args.size() != 1) throw EvalError(name + " needs one argument");
            return Value::str(args[0].listed());
        }
        if (name == "reshape") return reshape(args.at(0), positiveSize(args.at(1)), positiveSize(args.at(2)));
        if (name == "abs") return applyAbs(args.at(0));
        if (name == "sqrt") return applySqrt(args.at(0));
        if (name == "cudaversion") return Value::str(Engine::instance().cudaVersion());
        if (name == "rng") {
            if (args.empty()) return Value::scalar(DType::F64, static_cast<double>(session_.rngSeed));
            seedRng(session_, static_cast<std::uint64_t>(asNumber(args[0])));
            return Value::scalar(DType::F64, static_cast<double>(session_.rngSeed));
        }
        if (name == "expect") {
            if (args.size() < 2) throw EvalError("expect needs an expected value and an actual value");
            return expect(args[0], args[1], args.size() >= 3 ? asNumber(args[2]) : -1);
        }
        if (name == "diag") return diag(args.at(0));
        if (name == "linspace") {
            const auto count = args.size() >= 3 ? positiveSize(args[2]) : static_cast<std::int64_t>(100);
            return linspace(asNumber(args.at(0)), asNumber(args.at(1)), count);
        }
        if (name == "repmat") {
            const auto rows = positiveSize(args.at(1));
            const auto cols = args.size() >= 3 ? positiveSize(args[2]) : rows;
            return repmat(args.at(0), rows, cols);
        }
        if (name == "tril") return tril(args.at(0), args.size() >= 2 ? static_cast<std::int64_t>(asNumber(args[1])) : 0);
        if (name == "triu") return triu(args.at(0), args.size() >= 2 ? static_cast<std::int64_t>(asNumber(args[1])) : 0);
        if (name == "horzcat") return horzcat(args);
        if (name == "vertcat") return vertcat(args);
        if (name == "sign") return applySign(args.at(0));
        if (name == "exp" || name == "log" || name == "log2" || name == "log10" || name == "sin" || name == "cos" || name == "tan" ||
            name == "asin" || name == "acos" || name == "atan" || name == "sinh" || name == "cosh" || name == "tanh" || name == "floor" ||
            name == "ceil" || name == "round" || name == "fix") {
            int op = 3;
            if (name == "log") op = 4;
            else if (name == "log2") op = 5;
            else if (name == "log10") op = 6;
            else if (name == "sin") op = 7;
            else if (name == "cos") op = 8;
            else if (name == "tan") op = 9;
            else if (name == "asin") op = 10;
            else if (name == "acos") op = 11;
            else if (name == "atan") op = 12;
            else if (name == "sinh") op = 13;
            else if (name == "cosh") op = 14;
            else if (name == "tanh") op = 15;
            else if (name == "floor") op = 16;
            else if (name == "ceil") op = 17;
            else if (name == "round") op = 18;
            else if (name == "fix") op = 19;
            return applyUnary(args.at(0), op);
        }
        if (name == "atan2") return applyBinaryMath(args.at(0), args.at(1), 1);
        if (name == "mod") return applyBinaryMath(args.at(0), args.at(1), 2);
        if (name == "rem") return applyBinaryMath(args.at(0), args.at(1), 3);
        if (name == "power") return applyBinaryMath(args.at(0), args.at(1), 0);
        if (name == "sum" || name == "prod" || name == "any" || name == "all" || name == "mean") {
            const bool every = args.size() >= 2 && args.back().isString && args.back().text == "all";
            int dim = 1;
            if (!every && args.size() >= 2 && !args[1].isString && !args[1].empty()) dim = static_cast<int>(asNumber(args[1]));
            int op = 0;
            if (name == "prod") op = 1;
            if (name == "any") op = 4;
            if (name == "all") op = 5;
            Value reduced = reduce(args.at(0), op, dim, every);
            if (name != "mean") return reduced;
            const double denom = every ? static_cast<double>(args[0].numel()) : static_cast<double>(dim == 2 ? args[0].cols : args[0].rows);
            Value out = reduced.type == DType::F64 || reduced.type == DType::F32 ? reduced : convertTo(reduced, DType::F64);
            if (out.type != DType::F64 && out.type != DType::F32) out = convertTo(out, DType::F64);
            for (std::int64_t i = 0; i < out.numel(); ++i) out.set(i, denom == 0 ? 0 : out.get(i) / denom);
            return out;
        }
        if (name == "min" || name == "max") {
            if (args.size() >= 2 && !args[1].isString && !args[1].empty()) {
                return name == "min" ? minimum(args[0], args[1]) : maximum(args[0], args[1]);
            }
            int dim = 1;
            if (args.size() >= 3) dim = static_cast<int>(asNumber(args[2]));
            return reduce(args.at(0), name == "min" ? 2 : 3, dim, false);
        }
        if (name == "norm") {
            int kind = -1;
            if (args.size() >= 2) {
                if (args[1].isString && args[1].text == "fro") kind = 0;
                else if (args[1].isString && args[1].text == "inf") kind = 3;
                else {
                    const double p = asNumber(args[1]);
                    if (p == 1) kind = 1;
                    else if (std::isinf(p)) kind = 3;
                    else kind = 2;
                }
            }
            return norm(args.at(0), kind);
        }
        if (name == "dot") return dot(args.at(0), args.at(1));
        if (name == "cross") return cross(args.at(0), args.at(1));
        if (name == "trace") return trace(args.at(0));
        if (name == "randi") return randi(args);
        if (name == "inv") return inverse(args.at(0));
        if (name == "det") return determinant(args.at(0));
        if (name == "rank") return matrixRank(args.at(0));
        if (name == "chol") return chol(args.at(0));
        if (name == "lu" || name == "qr" || name == "svd" || name == "eig") {
            std::vector<Value> parts;
            if (name == "lu") parts = factorLu(args.at(0), nargout_);
            else if (name == "qr") parts = factorQr(args.at(0), nargout_);
            else if (name == "svd") parts = factorSvd(args.at(0), nargout_);
            else parts = factorEig(args.at(0), nargout_);
            multi_ = parts;
            return parts.empty() ? Value::matrix(DType::F64, 0, 0) : parts.front();
        }
        auto fn = session_.functions.find(name);
        if (fn != session_.functions.end()) return callUser(fn->second, args);
        throw EvalError("unknown function " + name);
    }

    Value expect(const Value& expected, const Value& actual, double tol) {
        const std::string name = session_.testName.empty() ? "expect" : session_.testName;
        auto fail = [&](const std::string& why) {
            if (session_.inSuite) ++session_.suiteFails;
            return Value::str("FAIL  " + name + ": " + why);
        };
        if (expected.isString || actual.isString) {
            if (expected.isString && actual.isString && expected.text == actual.text) {
                if (session_.inSuite) ++session_.suitePasses;
                return Value::str("PASS  " + name);
            }
            return fail(expected.isString && actual.isString ? "value mismatch" : "class mismatch");
        }
        if (expected.type != actual.type) return fail("class mismatch");
        if (expected.rows != actual.rows || expected.cols != actual.cols) return fail("size mismatch");
        double limit = tol;
        if (limit < 0) limit = (expected.type == DType::F32 || actual.type == DType::F32) ? 1e-5 : 1e-8;
        if (expected.type == DType::I32 || expected.type == DType::I64) limit = 0;
        double maxDiff = 0;
        for (std::int64_t i = 0; i < expected.numel(); ++i) {
            maxDiff = std::max(maxDiff, std::abs(expected.get(i) - actual.get(i)));
        }
        if (maxDiff > limit) {
            std::ostringstream why;
            why << "max abs diff " << maxDiff << ", tol " << limit;
            return fail(why.str());
        }
        if (session_.inSuite) ++session_.suitePasses;
        return Value::str("PASS  " + name);
    }

    Value callUser(const UserFunction& fn, const std::vector<Value>& args) {
        if (args.size() != fn.args.size()) throw EvalError(fn.name + " argument count");
        std::map<std::string, Value> locals;
        for (std::size_t n = 0; n < args.size(); ++n) locals[fn.args[n]] = args[n];
        Parser nested(session_, fn.body, Frame{&locals});
        nested.run();
        std::vector<Value> outs;
        for (const std::string& output : fn.outputs) {
            auto it = locals.find(output);
            if (it == locals.end()) throw EvalError(fn.name + " did not assign " + output);
            outs.push_back(it->second);
        }
        multi_ = outs;
        if (outs.empty()) return Value::matrix(DType::F64, 0, 0);
        return outs.front();
    }

    Value randi(const std::vector<Value>& args) {
        if (args.empty()) throw EvalError("randi needs a range");
        double low = 1;
        double high = 0;
        std::size_t at = 0;
        if (args[0].numel() == 2) {
            low = args[0].get(0);
            high = args[0].get(1);
            at = 1;
        } else {
            high = asNumber(args[0]);
            at = 1;
        }
        std::int64_t rows = 1;
        std::int64_t cols = 1;
        if (at < args.size() && !args[at].isString) {
            rows = positiveSize(args[at]);
            ++at;
            if (at < args.size() && !args[at].isString) {
                cols = positiveSize(args[at]);
                ++at;
            } else {
                cols = rows;
            }
        }
        DType type = DType::F64;
        if (at < args.size()) type = typeArg(args, at, DType::F64);
        Value drawn = randUniform(session_, rows, cols, DType::F64);
        const double span = high - low + 1;
        Value out = Value::matrix(type == DType::F32 ? DType::F64 : type, rows, cols);
        for (std::int64_t i = 0; i < drawn.numel(); ++i) {
            double sample = low + std::floor(drawn.get(i) * span);
            if (sample > high) sample = high;
            out.set(i, sample);
        }
        if (type == DType::F32) return convertTo(out, DType::F32);
        return out;
    }

    Value parsePrimary() {
        skip();
        if (eat('(')) {
            Value value = parseOr();
            if (!eat(')')) throw EvalError("expected )");
            return value;
        }
        if (eat('[')) return parseMatrix();
        if (peek() == '\'') return Value::str(quoted());
        if (std::isdigit(static_cast<unsigned char>(peek())) || peek() == '.') return parseNumber();
        const std::string name = ident();
        skip();
        if (indexDepth_ > 0 && name == "end") return Value::scalar(DType::F64, static_cast<double>(indexEnd_));
        Value* found = lookup(name);
        if (eat('(')) {
            if (found != nullptr && !found->isString) return postfix(parseIndex(*found));
            std::vector<Value> args;
            skip();
            if (!eat(')')) {
                do {
                    args.push_back(parseOr());
                    skip();
                } while (eat(','));
                if (!eat(')')) throw EvalError("expected )");
            }
            return postfix(call(name, args));
        }
        if (found == nullptr) throw EvalError("undefined variable " + name);
        return postfix(*found);
    }

    Value parseIndex(const Value& value) {
        bool two = false;
        int depth = 1;
        for (std::size_t pos = i_; pos < source_.size() && depth > 0; ++pos) {
            const char c = source_[pos];
            if (c == '(') ++depth;
            else if (c == ')') --depth;
            else if (c == ',' && depth == 1) two = true;
        }
        std::vector<Value> indexes;
        std::vector<bool> colon;
        do {
            skip();
            if (eat(':')) {
                colon.push_back(true);
                indexes.push_back(Value::matrix(DType::F64, 0, 0));
            } else {
                const int axis = static_cast<int>(indexes.size());
                const int savedDepth = indexDepth_;
                const auto savedEnd = indexEnd_;
                ++indexDepth_;
                indexEnd_ = !two ? value.numel() : (axis == 0 ? value.rows : value.cols);
                indexes.push_back(parseOr());
                colon.push_back(false);
                indexDepth_ = savedDepth;
                indexEnd_ = savedEnd;
            }
            skip();
        } while (eat(','));
        if (!eat(')')) throw EvalError("expected )");
        return subscript(value, indexes, colon);
    }

    Value postfix(Value value) {
        for (;;) {
            skip();
            if (peek() == '.' && i_ + 1 < source_.size() && source_[i_ + 1] == '\'') {
                i_ += 2;
                value = transpose(value);
                continue;
            }
            if (eat('\'')) {
                value = transpose(value);
                continue;
            }
            break;
        }
        return value;
    }

    Value parseMatrix() {
        std::vector<std::vector<Value>> rows;
        rows.push_back({});
        skip();
        if (eat(']')) return Value::matrix(DType::F64, 0, 0);
        while (true) {
            rows.back().push_back(parseOr());
            skip();
            if (eat(';')) {
                rows.push_back({});
                continue;
            }
            if (eat(']')) break;
            if (eat(',')) continue;
            if (done()) throw EvalError("expected ]");
        }
        if (!rows.empty() && rows.back().empty()) rows.pop_back();
        std::vector<Value> stacked;
        for (const auto& row : rows) stacked.push_back(horzcat(row));
        return vertcat(stacked);
    }

    Value parseNumber() {
        const std::size_t mark = i_;
        while (std::isdigit(static_cast<unsigned char>(peek()))) ++i_;
        if (peek() == '.') {
            ++i_;
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++i_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++i_;
            if (peek() == '+' || peek() == '-') ++i_;
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++i_;
        }
        return Value::scalar(DType::F64, std::stod(source_.substr(mark, i_ - mark)));
    }

    Value parseUnary() {
        skip();
        if (eat('+')) return parseUnary();
        if (eat('-')) return unaryMinus(parseUnary());
        if (eat('~')) return logicalNot(parseUnary());
        return parsePrimary();
    }

    Value parseMul() {
        Value left = parseUnary();
        for (;;) {
            skip();
            char op = 0;
            bool element = false;
            if (eat('.') && (peek() == '*' || peek() == '/' || peek() == '^')) {
                element = peek() != '^';
                op = peek();
                ++i_;
            } else if (peek() == '*' || peek() == '/' || peek() == '\\' || peek() == '^') {
                op = peek();
                ++i_;
            } else break;
            Value right = parseUnary();
            if (op == '^') left = applyBinaryMath(left, right, 0);
            else if (op == '\\') left = mldivide(left, right);
            else if (op == '/' && !element) left = mrdivide(left, right);
            else left = binary(left, right, op, element);
        }
        return left;
    }

    Value parseAdd() {
        Value left = parseMul();
        for (;;) {
            skip();
            char op = 0;
            if (peek() == '+' || peek() == '-') {
                op = peek();
                ++i_;
            } else break;
            left = binary(left, parseMul(), op, true);
        }
        return left;
    }

    Value parseColon() {
        Value left = parseAdd();
        if (!eat(':')) return left;
        Value second = parseAdd();
        if (!eat(':')) return colon(asNumber(left), 1, asNumber(second));
        return colon(asNumber(left), asNumber(second), asNumber(parseAdd()));
    }

    Value colon(double start, double step, double stop) {
        if (step == 0) throw EvalError("colon step must not be zero");
        std::vector<double> numbers;
        if (step > 0) {
            for (double v = start; v <= stop + step * 1e-12; v += step) numbers.push_back(v);
        } else {
            for (double v = start; v >= stop + step * 1e-12; v += step) numbers.push_back(v);
        }
        Value out = Value::matrix(DType::F64, 1, static_cast<std::int64_t>(numbers.size()));
        for (std::size_t n = 0; n < numbers.size(); ++n) out.set(static_cast<std::int64_t>(n), numbers[n]);
        return out;
    }

    Value parseCompare() {
        Value left = parseColon();
        skip();
        const char first = peek();
        if (first != '=' && first != '~' && first != '<' && first != '>') return left;
        const std::size_t mark = i_;
        ++i_;
        std::string op(1, first);
        if (peek() == '=') {
            op.push_back('=');
            ++i_;
        } else if (first == '=' || first == '~') {
            i_ = mark;
            return left;
        }
        return compare(left, parseColon(), op.c_str());
    }

    Value parseAnd() {
        Value left = parseCompare();
        for (;;) {
            skip();
            if (!eat('&')) return left;
            if (eat('&')) {
                Value right = parseCompare();
                left = Value::scalar(DType::F64, isTrue(left) && isTrue(right) ? 1 : 0);
            } else {
                left = logicalAnd(left, parseCompare());
            }
        }
    }

    Value parseOr() {
        Value left = parseAnd();
        for (;;) {
            skip();
            if (!eat('|')) return left;
            if (eat('|')) {
                const bool decided = isTrue(left);
                Value right = parseAnd();
                if (decided) left = Value::scalar(DType::F64, 1);
                else left = Value::scalar(DType::F64, isTrue(right) ? 1 : 0);
            } else {
                left = logicalOr(left, parseAnd());
            }
        }
    }

    static bool opensBlock(const std::string& word) {
        return word == "function" || word == "if" || word == "for" || word == "while" || word == "test" || word == "tests";
    }

    std::string readWord() {
        std::string word;
        while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_') {
            char c = peek();
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            word.push_back(c);
            ++i_;
        }
        return word;
    }

    std::string grabUntilEnd(bool stopClause = false) {
        int depth = 1;
        int nest = 0;
        std::string body;
        while (!done() && depth > 0) {
            if (nest == 0 && (peek() == '(' || peek() == '[')) {
                body.push_back(peek());
                ++i_;
                ++nest;
                continue;
            }
            if (nest > 0 && (peek() == '(' || peek() == '[')) {
                body.push_back(peek());
                ++i_;
                ++nest;
                continue;
            }
            if (nest > 0 && (peek() == ')' || peek() == ']')) {
                body.push_back(peek());
                ++i_;
                --nest;
                continue;
            }
            if (peek() == '\'') {
                const char previous = i_ == 0 ? '\0' : source_[i_ - 1];
                const bool transpose = !std::isspace(static_cast<unsigned char>(previous)) &&
                    (std::isalnum(static_cast<unsigned char>(previous)) || previous == '_' || previous == ')' || previous == ']' || previous == '.');
                body.push_back('\'');
                ++i_;
                if (!transpose) {
                    while (!done() && peek() != '\'') {
                        body.push_back(peek());
                        ++i_;
                    }
                    if (!done()) {
                        body.push_back(peek());
                        ++i_;
                    }
                }
                continue;
            }
            if (peek() == '%') {
                while (!done() && peek() != '\n') {
                    body.push_back(peek());
                    ++i_;
                }
                continue;
            }
            if (std::isalpha(static_cast<unsigned char>(peek()))) {
                const std::size_t mark = i_;
                const std::string word = readWord();
                const char after = peek();
                const bool boundary = nest == 0 && !std::isalnum(static_cast<unsigned char>(after)) && after != '_';
                if (boundary && stopClause && depth == 1 && (word == "else" || word == "elseif")) {
                    i_ = mark;
                    return body;
                }
                if (boundary && word == "end") {
                    --depth;
                    if (depth == 0) {
                        const std::size_t afterEnd = i_;
                        skip();
                        if (std::isalpha(static_cast<unsigned char>(peek()))) {
                            const std::size_t testsAt = i_;
                            if (readWord() == "tests") return body;
                            i_ = testsAt;
                        }
                        i_ = afterEnd;
                        return body;
                    }
                    body += "end";
                    continue;
                }
                if (boundary && opensBlock(word)) ++depth;
                body.append(source_, mark, i_ - mark);
                continue;
            }
            body.push_back(peek());
            ++i_;
        }
        return body;
    }

    std::string statement(std::ostringstream& printed) {
        skip();
        if (peek() == '?') {
            ++i_;
            return answerText(helpText());
        }
        if (eatWord("help")) return answerText(helpText());
        if (eatWord("who")) return answerText(who());
        if (eatWord("whos")) return answerText(whos());
        if (eatWord("clear")) return clearCommand();
        if (eatWord("clc")) return "\x01";
        if (eatWord("exit") || eatWord("quit")) return "\x02";
        if (eatWord("timer")) return answerText(timerCommand());
        if (eatWord("memory")) {
            MemoryInfo info = Engine::instance().memory(session_);
            assign("ans", Value::matrix(DType::F64, 3, 1));
            session_.vars["ans"].set(0, info.workspaceBytes);
            session_.vars["ans"].set(1, info.cudaTotal);
            session_.vars["ans"].set(2, info.cudaFree);
            return info.text;
        }
        if (eatWord("cudaversion")) {
            Value value = Value::str(Engine::instance().cudaVersion());
            const std::string text = value.text;
            assign("ans", value);
            return text;
        }
        if (eatWord("saveworkspace")) return saveWorkspace();
        if (eatWord("loadworkspace")) return loadWorkspace();
        {
            const std::size_t at = i_;
            if (eatWord("save") && eat('(')) return saveData();
            i_ = at;
            if (eatWord("load") && eat('(')) return loadData();
            i_ = at;
        }
        if (eatWord("runtests")) return runFile(true);
        if (eatWord("run")) return runFile(false);
        if (eatWord("tests")) return testsBlock();
        if (eatWord("test")) return testBlock(printed);
        if (eatWord("if")) return ifBlock(printed);
        if (eatWord("for")) return forBlock(printed);
        if (eatWord("while")) return whileBlock(printed);
        if (eatWord("function")) return functionDef();
        if (eat('[')) {
            std::vector<std::string> names;
            skip();
            if (!eat(']')) {
                do {
                    skip();
                    names.push_back(ident());
                    skip();
                } while (eat(','));
                if (!eat(']')) throw EvalError("expected ]");
            }
            if (!eat('=')) throw EvalError("expected =");
            const int saved = nargout_;
            nargout_ = static_cast<int>(names.size());
            Value value = parseOr();
            if (multi_.size() != names.size()) {
                if (names.size() != 1) throw EvalError("output count");
                assign(names[0], value);
            } else {
                for (std::size_t n = 0; n < names.size(); ++n) assign(names[n], multi_[n]);
                value = multi_.front();
            }
            multi_.clear();
            nargout_ = saved;
            assign("ans", value);
            return value.format();
        }
        const std::size_t mark = i_;
        if (std::isalpha(static_cast<unsigned char>(peek())) || peek() == '_') {
            const std::string name = ident();
            skip();
            if (eat('=')) {
                Value value = parseOr();
                Value shown = value;
                assign(name, std::move(value));
                assign("ans", shown);
                return shown.format();
            }
            i_ = mark;
        }
        Value value = parseOr();
        if (!frame_.locals) assign("ans", value);
        return value.format();
    }

    std::string answerText(std::string text) {
        assign("ans", Value::str(text));
        return text;
    }

    std::string who() const {
        std::ostringstream out;
        for (const auto& entry : session_.vars) out << entry.first << '\n';
        return out.str();
    }

    std::string whos() const {
        std::ostringstream out;
        for (const auto& entry : session_.vars) out << entry.second.whosLine(entry.first) << '\n';
        for (const auto& entry : session_.functions) out << entry.first << "  function\n";
        return out.str();
    }

    std::string restOfLine() {
        std::string word;
        while (peek() == ' ' || peek() == '\t') ++i_;
        while (std::isalpha(static_cast<unsigned char>(peek()))) {
            char c = peek();
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            word.push_back(c);
            ++i_;
        }
        return word;
    }

    std::string clearCommand() {
        const std::string word = restOfLine();
        if (word == "functions") {
            session_.functions.clear();
            return {};
        }
        if (word.empty()) {
            session_.vars.clear();
            session_.functions.clear();
            session_.dirty = true;
            return {};
        }
        session_.vars.erase(word);
        session_.functions.erase(word);
        session_.dirty = true;
        return {};
    }

    std::string timerCommand() {
        const std::string word = restOfLine();
        if (word == "on") session_.timer = true;
        else if (word == "off") session_.timer = false;
        return session_.timer ? "timer on" : "timer off";
    }

    bool dataPath(const std::string& path) const {
        if (path.size() < 4) return false;
        std::string ext = path.substr(path.size() - 4);
        for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return ext == ".cma";
    }

    void writeOneVariable(std::ostream& file, std::ofstream& sidecar, std::uint64_t& sidecarAt, const std::string& path,
                          const std::string& name, const Value& value) {
        const auto bytes = static_cast<std::uint64_t>(value.numel()) * dtypeSize(value.type);
        file << "{\"name\":\"" << name << "\",\"type\":\"" << dtypeName(value.type)
             << "\",\"rows\":" << value.rows << ",\"cols\":" << value.cols << ',';
        if (bytes > 1024ull * 1024ull) {
            if (!sidecar.is_open()) {
                sidecar.open(path + ".bin", std::ios::binary);
                if (!sidecar) throw EvalError("could not write " + path + ".bin");
            }
            sidecar.write(static_cast<const char*>(value.data()), static_cast<std::streamsize>(bytes));
            file << "\"sidecar\":{\"offset\":" << sidecarAt << ",\"length\":" << bytes << "}}";
            sidecarAt += bytes;
        } else {
            file << "\"values\":[";
            for (std::int64_t n = 0; n < value.numel(); ++n) {
                if (n) file << ',';
                file << value.get(n);
            }
            file << "]}";
        }
    }

    std::string saveData() {
        const std::string path = pathArgument();
        if (!dataPath(path)) throw EvalError("save writes a .cma data file");
        std::vector<std::string> names;
        skip();
        if (!eat(',')) throw EvalError("save needs variable names");
        do {
            names.push_back(ident());
            skip();
        } while (eat(','));
        if (!eat(')')) throw EvalError("expected )");
        std::vector<std::pair<std::string, Value>> saved;
        for (const std::string& name : names) {
            Value* found = lookup(name);
            if (found == nullptr || found->isString) throw EvalError("undefined variable " + name);
            saved.emplace_back(name, *found);
        }
        std::ofstream file(path, std::ios::binary);
        if (!file) throw EvalError("could not write " + path);
        file << "{\"version\":1,\"kind\":\"variables\",\"variables\":[";
        std::ofstream sidecar;
        std::uint64_t sidecarAt = 0;
        for (std::size_t n = 0; n < saved.size(); ++n) {
            if (n) file << ',';
            writeOneVariable(file, sidecar, sidecarAt, path, saved[n].first, saved[n].second);
        }
        file << "]}";
        return {};
    }

    std::string loadData() {
        const std::string path = pathArgument();
        if (!dataPath(path)) throw EvalError("load reads a .cma data file");
        std::vector<std::string> only;
        skip();
        if (eat(',')) {
            do {
                only.push_back(ident());
                skip();
            } while (eat(','));
        }
        if (!eat(')')) throw EvalError("expected )");
        std::ifstream file(path, std::ios::binary);
        if (!file) throw EvalError("could not read " + path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        const std::string text = buffer.str();
        if (text.find("\"kind\":\"variables\"") == std::string::npos) throw EvalError("not a .cma data file");
        auto findNumber = [&](const std::string& key, std::size_t from) {
            auto pos = text.find("\"" + key + "\":", from);
            if (pos == std::string::npos) return 0.0;
            pos = text.find_first_of("-0123456789", pos);
            return std::stod(text.substr(pos));
        };
        std::map<std::string, Value> loaded;
        std::size_t cursor = 0;
        while ((cursor = text.find("\"name\":\"", cursor)) != std::string::npos) {
            cursor += 8;
            const auto nameEnd = text.find('"', cursor);
            const std::string name = text.substr(cursor, nameEnd - cursor);
            const auto nextName = text.find("\"name\":\"", nameEnd);
            const std::size_t objectEnd = nextName == std::string::npos ? text.size() : nextName;
            const auto typePos = text.find("\"type\":\"", nameEnd);
            if (typePos == std::string::npos || typePos >= objectEnd) {
                cursor = nameEnd;
                continue;
            }
            const auto typeStart = typePos + 8;
            const auto typeEnd = text.find('"', typeStart);
            const std::string typeName = text.substr(typeStart, typeEnd - typeStart);
            DType type = DType::F64;
            if (typeName == "single") type = DType::F32;
            if (typeName == "int32") type = DType::I32;
            if (typeName == "int64") type = DType::I64;
            const auto rows = static_cast<std::int64_t>(findNumber("rows", typeEnd));
            const auto cols = static_cast<std::int64_t>(findNumber("cols", typeEnd));
            Value value = Value::matrix(type, rows, cols);
            const auto side = text.find("\"sidecar\"", typeEnd);
            if (side != std::string::npos && side < objectEnd) {
                const auto offset = static_cast<std::uint64_t>(findNumber("offset", side));
                const auto length = static_cast<std::uint64_t>(findNumber("length", side));
                std::ifstream bin(path + ".bin", std::ios::binary);
                if (!bin) throw EvalError("could not read " + path + ".bin");
                bin.seekg(static_cast<std::streamoff>(offset));
                bin.read(static_cast<char*>(value.data()), static_cast<std::streamsize>(length));
                if (!bin) throw EvalError("could not read " + path + ".bin");
            } else {
                auto values = text.find("\"values\":[", typeEnd);
                values += 10;
                for (std::int64_t n = 0; n < value.numel(); ++n) {
                    values = text.find_first_of("-0123456789", values);
                    std::size_t stop = values;
                    while (stop < text.size() && (std::isdigit(static_cast<unsigned char>(text[stop])) || text[stop] == '-' || text[stop] == '.' || text[stop] == 'e' || text[stop] == 'E' || text[stop] == '+')) ++stop;
                    value.set(n, std::stod(text.substr(values, stop - values)));
                    values = stop;
                }
            }
            loaded.emplace(name, std::move(value));
            cursor = nameEnd;
        }
        const std::vector<std::string> chosen = only.empty() ? [&] {
            std::vector<std::string> all;
            for (const auto& entry : loaded) all.push_back(entry.first);
            return all;
        }() : only;
        for (const std::string& name : chosen) {
            auto it = loaded.find(name);
            if (it == loaded.end()) throw EvalError("data file has no variable " + name);
            assign(name, it->second);
        }
        return {};
    }

    std::string pathArgument() {
        skip();
        if (peek() == '\'') return quoted();
        std::string path;
        while (!done() && !std::isspace(static_cast<unsigned char>(peek())) && peek() != ';') {
            path.push_back(peek());
            ++i_;
        }
        if (path.empty()) throw EvalError("expected a path");
        return path;
    }

    std::string saveWorkspace() {
        const std::string path = pathArgument();
        std::ofstream file(path, std::ios::binary);
        if (!file) throw EvalError("could not write " + path);
        file << "{\"version\":1,\"seed\":" << session_.rngSeed << ",\"drawn\":" << session_.rngDrawn << ",\"variables\":[";
        std::ofstream sidecar;
        std::uint64_t sidecarAt = 0;
        bool first = true;
        for (const auto& entry : session_.vars) {
            if (entry.second.isString) continue;
            if (!first) file << ',';
            first = false;
            const auto bytes = static_cast<std::uint64_t>(entry.second.numel()) * dtypeSize(entry.second.type);
            file << "{\"name\":\"" << entry.first << "\",\"type\":\"" << dtypeName(entry.second.type)
                 << "\",\"rows\":" << entry.second.rows << ",\"cols\":" << entry.second.cols << ',';
            if (bytes > 1024ull * 1024ull) {
                if (!sidecar.is_open()) {
                    sidecar.open(path + ".bin", std::ios::binary);
                    if (!sidecar) throw EvalError("could not write " + path + ".bin");
                }
                sidecar.write(static_cast<const char*>(entry.second.data()), static_cast<std::streamsize>(bytes));
                file << "\"sidecar\":{\"offset\":" << sidecarAt << ",\"length\":" << bytes << "}}";
                sidecarAt += bytes;
            } else {
                file << "\"values\":[";
                for (std::int64_t n = 0; n < entry.second.numel(); ++n) {
                    if (n) file << ',';
                    file << entry.second.get(n);
                }
                file << "]}";
            }
        }
        file << "],\"functions\":[";
        first = true;
        for (const auto& entry : session_.functions) {
            if (!first) file << ',';
            first = false;
            file << "{\"name\":\"" << entry.second.name << "\",\"outputs\":[";
            for (std::size_t n = 0; n < entry.second.outputs.size(); ++n) {
                if (n) file << ',';
                file << '"' << entry.second.outputs[n] << '"';
            }
            file << "],\"args\":[";
            for (std::size_t n = 0; n < entry.second.args.size(); ++n) {
                if (n) file << ',';
                file << '"' << entry.second.args[n] << '"';
            }
            file << "],\"body\":\"";
            for (char c : entry.second.body) {
                if (c == '"' || c == '\\') file << '\\';
                if (c == '\n') file << "\\n";
                else file << c;
            }
            file << "\"}";
        }
        file << "]}";
        session_.path = path;
        session_.dirty = false;
        return {};
    }

    std::string loadWorkspace() {
        const std::string path = pathArgument();
        std::ifstream file(path, std::ios::binary);
        if (!file) throw EvalError("could not read " + path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        const std::string text = buffer.str();
        session_.vars.clear();
        session_.functions.clear();
        session_.resetRng();
        auto findNumber = [&](const std::string& key, std::size_t from) {
            auto pos = text.find("\"" + key + "\":", from);
            if (pos == std::string::npos) return 0.0;
            pos = text.find_first_of("-0123456789", pos);
            return std::stod(text.substr(pos));
        };
        session_.rngSeed = static_cast<std::uint64_t>(findNumber("seed", 0));
        session_.rngDrawn = static_cast<std::uint64_t>(findNumber("drawn", 0));
        auto jsonString = [&](const std::string& key, std::size_t from, std::size_t until) {
            const auto keyAt = text.find("\"" + key + "\":\"", from);
            if (keyAt == std::string::npos || keyAt >= until) return std::string{};
            std::size_t pos = keyAt + key.size() + 4;
            std::string out;
            while (pos < until && text[pos] != '"') {
                if (text[pos] == '\\' && pos + 1 < until) {
                    const char next = text[++pos];
                    out.push_back(next == 'n' ? '\n' : next);
                } else {
                    out.push_back(text[pos]);
                }
                ++pos;
            }
            return out;
        };
        auto jsonNames = [&](const std::string& key, std::size_t from, std::size_t until) {
            std::vector<std::string> names;
            const auto keyAt = text.find("\"" + key + "\":[", from);
            if (keyAt == std::string::npos || keyAt >= until) return names;
            std::size_t pos = keyAt + key.size() + 3;
            const auto end = text.find(']', pos);
            if (end == std::string::npos || end > until) return names;
            while (pos < end) {
                const auto quote = text.find('"', pos);
                if (quote == std::string::npos || quote >= end) break;
                const auto close = text.find('"', quote + 1);
                if (close == std::string::npos || close > end) break;
                names.push_back(text.substr(quote + 1, close - quote - 1));
                pos = close + 1;
            }
            return names;
        };
        std::size_t cursor = 0;
        while ((cursor = text.find("\"name\":\"", cursor)) != std::string::npos) {
            cursor += 8;
            const auto nameEnd = text.find('"', cursor);
            const std::string name = text.substr(cursor, nameEnd - cursor);
            const auto nextName = text.find("\"name\":\"", nameEnd);
            const std::size_t objectEnd = nextName == std::string::npos ? text.size() : nextName;
            const auto typePos = text.find("\"type\":\"", nameEnd);
            const bool isFunction = text.find("\"body\"", nameEnd) < text.find("\"values\"", nameEnd);
            if (isFunction) {
                UserFunction fn;
                fn.name = name;
                fn.outputs = jsonNames("outputs", nameEnd, objectEnd);
                fn.args = jsonNames("args", nameEnd, objectEnd);
                fn.body = jsonString("body", nameEnd, objectEnd);
                session_.functions[fn.name] = std::move(fn);
            } else if (typePos != std::string::npos && typePos < objectEnd) {
                const auto typeStart = typePos + 8;
                const auto typeEnd = text.find('"', typeStart);
                const std::string typeName = text.substr(typeStart, typeEnd - typeStart);
                DType type = DType::F64;
                if (typeName == "single") type = DType::F32;
                if (typeName == "int32") type = DType::I32;
                if (typeName == "int64") type = DType::I64;
                const auto rows = static_cast<std::int64_t>(findNumber("rows", typeEnd));
                const auto cols = static_cast<std::int64_t>(findNumber("cols", typeEnd));
                Value value = Value::matrix(type, rows, cols);
                const auto side = text.find("\"sidecar\"", typeEnd);
                if (side != std::string::npos && side < objectEnd) {
                    const auto offset = static_cast<std::uint64_t>(findNumber("offset", side));
                    const auto length = static_cast<std::uint64_t>(findNumber("length", side));
                    std::ifstream bin(path + ".bin", std::ios::binary);
                    if (!bin) throw EvalError("could not read " + path + ".bin");
                    bin.seekg(static_cast<std::streamoff>(offset));
                    bin.read(static_cast<char*>(value.data()), static_cast<std::streamsize>(length));
                    if (!bin) throw EvalError("could not read " + path + ".bin");
                } else {
                    auto values = text.find("\"values\":[", typeEnd);
                    values += 10;
                    for (std::int64_t n = 0; n < value.numel(); ++n) {
                        values = text.find_first_of("-0123456789", values);
                        std::size_t stop = values;
                        while (stop < text.size() && (std::isdigit(static_cast<unsigned char>(text[stop])) || text[stop] == '-' || text[stop] == '.' || text[stop] == 'e' || text[stop] == 'E' || text[stop] == '+')) ++stop;
                        value.set(n, std::stod(text.substr(values, stop - values)));
                        values = stop;
                    }
                }
                session_.vars[name] = std::move(value);
            }
            cursor = nameEnd;
        }
        session_.path = path;
        session_.dirty = false;
        return {};
    }

    std::string runFile(bool suite) {
        const std::string path = pathArgument();
        std::ifstream file(path, std::ios::binary);
        if (!file) throw EvalError("could not read " + path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        const std::string source = buffer.str();
        const bool hasSuite = source.find("tests") != std::string::npos && source.find("end tests") != std::string::npos;
        if (suite && !hasSuite) throw EvalError("runtests requires a tests/end tests suite");
        if (!suite && hasSuite) {
            return runFileSaved(source, true);
        }
        if (suite) return runFileSaved(source, true);
        Parser nested(session_, source, frame_);
        return nested.run();
    }

    std::string runFileSaved(const std::string& source, bool) {
        auto vars = session_.vars;
        auto fns = session_.functions;
        const auto seed = session_.rngSeed;
        const auto drawn = session_.rngDrawn;
        session_.vars.clear();
        session_.functions.clear();
        session_.resetRng();
        session_.rngSeed = 0;
        session_.rngDrawn = 0;
        session_.suitePasses = 0;
        session_.suiteFails = 0;
        std::string output;
        try {
            Parser nested(session_, source, Frame{});
            output = nested.run();
        } catch (...) {
            session_.vars = std::move(vars);
            session_.functions = std::move(fns);
            session_.resetRng();
            session_.rngSeed = seed;
            session_.rngDrawn = drawn;
            throw;
        }
        const int fails = session_.suiteFails;
        session_.vars = std::move(vars);
        session_.functions = std::move(fns);
        session_.resetRng();
        session_.rngSeed = seed;
        session_.rngDrawn = drawn;
        assign("ans", Value::scalar(DType::F64, fails));
        return output;
    }

    std::string testsBlock() {
        skip();
        session_.suiteName = peek() == '\'' ? quoted() : ident();
        session_.inSuite = true;
        session_.suitePasses = 0;
        session_.suiteFails = 0;
        const auto start = std::chrono::steady_clock::now();
        std::ostringstream out;
        const std::string body = grabUntilEnd();
        Parser nested(session_, body, frame_);
        out << nested.run();
        eatWord("tests");
        session_.inSuite = false;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        out << session_.suiteName << "\nElapsed " << ms << " ms\n" << session_.suitePasses << " passed, " << session_.suiteFails << " failed\n";
        return out.str();
    }

    std::string testBlock(std::ostringstream&) {
        skip();
        session_.testName = peek() == '\'' ? quoted() : ident();
        const std::string body = grabUntilEnd();
        Parser nested(session_, body, frame_);
        const std::string output = nested.run();
        session_.testName.clear();
        return output;
    }

    std::string ifBlock(std::ostringstream&) {
        std::string chosen;
        bool taken = false;
        while (true) {
            Value cond = parseOr();
            const std::string body = grabUntilEnd(true);
            if (!taken && isTrue(cond)) {
                Parser nested(session_, body, frame_);
                chosen = nested.run();
                taken = true;
            }
            skip();
            if (eatWord("elseif")) continue;
            if (eatWord("else")) {
                const std::string rest = grabUntilEnd(false);
                if (!taken) {
                    Parser nested(session_, rest, frame_);
                    return nested.run();
                }
                return chosen;
            }
            return chosen;
        }
    }

    std::string forBlock(std::ostringstream&) {
        const std::string loopName = ident();
        if (!eat('=')) throw EvalError("expected =");
        Value values = parseOr();
        const std::string body = grabUntilEnd();
        std::ostringstream out;
        for (std::int64_t n = 0; n < values.numel(); ++n) {
            assign(loopName, Value::scalar(values.type, values.get(n)));
            Parser nested(session_, body, frame_);
            out << nested.run();
        }
        return out.str();
    }

    std::string whileBlock(std::ostringstream&) {
        const std::size_t condAt = i_;
        Value cond = parseOr();
        const std::size_t afterCond = i_;
        const std::string body = grabUntilEnd();
        std::ostringstream out;
        int guard = 0;
        while (isTrue(cond)) {
            Parser nested(session_, body, frame_);
            out << nested.run();
            i_ = condAt;
            cond = parseOr();
            i_ = afterCond;
            if (++guard > 1000000) throw EvalError("while did not finish");
        }
        i_ = source_.size();
        return out.str();
    }

    std::string functionDef() {
        skip();
        UserFunction fn;
        if (eat('[')) {
            skip();
            if (!eat(']')) {
                do {
                    skip();
                    fn.outputs.push_back(ident());
                    skip();
                } while (eat(','));
                if (!eat(']')) throw EvalError("expected ]");
            }
            if (!eat('=')) throw EvalError("expected =");
            fn.name = ident();
        } else if (std::isalpha(static_cast<unsigned char>(peek())) || peek() == '_') {
            const std::string first = ident();
            skip();
            if (eat('=')) {
                fn.outputs.push_back(first);
            } else {
                fn.name = first;
            }
        }
        if (fn.name.empty()) fn.name = ident();
        if (eat('(')) {
            skip();
            if (!eat(')')) {
                do {
                    fn.args.push_back(ident());
                } while (eat(','));
                if (!eat(')')) throw EvalError("expected )");
            }
        }
        fn.body = grabUntilEnd();
        session_.functions[fn.name] = fn;
        session_.dirty = true;
        return {};
    }
};

ExecResult executeUnlocked(Session& session, std::string_view input) {
    ExecResult result;
    if (submissionIsOpen(input)) {
        result.incomplete = true;
        return result;
    }
    try {
        Parser parser(session, std::string(input), Frame{});
        std::string output = parser.run();
        if (output == "\x01") {
            result.clearOutput = true;
            return result;
        }
        if (output == "\x02") {
            result.requestExit = true;
            return result;
        }
        if (!output.empty() && output[0] == '\x01') {
            result.clearOutput = true;
            output.erase(output.begin());
        }
        result.output = std::move(output);
    } catch (const std::exception& error) {
        result.output = std::string("Error: ") + error.what() + "\n";
    }
    return result;
}

}  // namespace

ExecResult execute(Engine& engine, Session& session, std::string_view input) {
    std::lock_guard<std::mutex> lock(engine.submissionMutex());
    if (submissionIsOpen(input)) {
        ExecResult pending;
        pending.incomplete = true;
        return pending;
    }
    const auto start = std::chrono::steady_clock::now();
    ExecResult result = executeUnlocked(session, input);
    cudaDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::size_t timerAt = 0;
    while (timerAt < input.size() && std::isspace(static_cast<unsigned char>(input[timerAt]))) ++timerAt;
    const bool timerCommand = input.substr(timerAt).starts_with("timer") &&
        (timerAt + 5 >= input.size() || !std::isalnum(static_cast<unsigned char>(input[timerAt + 5])));
    if (!timerCommand) session.lastElapsedMs = ms;
    result.elapsedMs = ms;
    result.showElapsed = session.timer && !result.incomplete && !timerCommand;
    if (result.showElapsed) {
        std::ostringstream line;
        line.setf(std::ios::fixed);
        line.precision(3);
        line << "Elapsed " << ms << " ms\n";
        result.output += line.str();
    }
    return result;
}

}  // namespace cudamath
