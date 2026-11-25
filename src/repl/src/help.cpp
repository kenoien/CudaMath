#include "cudamath/repl.hpp"

#include <sstream>

namespace cudamath {

std::string helpText() {
    return
        "Commands\n"
        "  help                 commands, types, and functions\n"
        "  who                  list defined names\n"
        "  whos                 name, type, and size\n"
        "  clear                clear one name, or the workspace\n"
        "  clear functions      remove user functions\n"
        "  saveworkspace path   write a .cmw workspace file\n"
        "  loadworkspace path   replace the workspace from a .cmw file\n"
        "  run path             execute a .cm script in this session\n"
        "  clc                  clear the REPL output\n"
        "  expect               compare an actual value with an expected value\n"
        "  runtests path        run a .cm suite in a scratch workspace\n"
        "  timer on|off         report elapsed time after each submission\n"
        "  timer                print timer on or timer off\n"
        "  cudaversion          CUDA runtime version\n"
        "  memory               workspace bytes, CUDA total, and CUDA free\n"
        "  exit, quit           leave the console process\n"
        "\n"
        "Types\n"
        "  double, single, int32, int64\n"
        "  A bare literal is double. Values are dense 2-D column-major arrays. A scalar is 1-by-1.\n"
        "  There is no logical type. A condition is true when it is nonempty and every element is nonzero.\n"
        "\n"
        "Functions\n"
        "  zeros, ones, eye, size, numel, length, ndims, isempty, class\n"
        "  double, single, int32, int64, reshape\n"
        "  rng, rand, randn\n"
        "  abs, sqrt, sum\n"
        "  Operators + - .* ./ * ' and comparisons\n"
        "  function/end, if/end, for/end, while/end\n"
        "  tests/end tests and test/end group expect checks\n"
        "\n"
        "Workspace files use .cmw. Scripts and suites use .cm.\n";
}

std::string statusWorkspace(const Session& session) {
    return session.title();
}

std::string statusTimer(const Session& session) {
    std::ostringstream out;
    out << (session.timer ? "Timer on" : "Timer off");
    if (session.timer) {
        out.setf(std::ios::fixed);
        out.precision(3);
        out << "  " << session.lastElapsedMs << " ms";
    }
    return out.str();
}

std::string statusMemory(const Session& session) {
    const MemoryInfo info = Engine::instance().memory(session);
    const double mb = 1024.0 * 1024.0;
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    out << "Workspace used " << (info.workspaceBytes / mb) << " MB    CUDA Free " << (info.cudaFree / mb) << " MB";
    return out.str();
}

bool submissionIsOpen(std::string_view input) {
    int depth = 0;
    std::string word;
    std::string previous;
    auto flush = [&]() {
        if (word.empty()) return;
        if (word == "tests" && previous == "end") {
            previous.clear();
        } else if (word == "function" || word == "if" || word == "for" || word == "while" || word == "test" || word == "tests") {
            ++depth;
            previous = word;
        } else if (word == "end") {
            if (depth > 0) --depth;
            previous = "end";
        } else {
            previous.clear();
        }
        word.clear();
    };
    bool comment = false;
    bool quote = false;
    for (char c : input) {
        if (c == '\n') comment = false;
        if (!quote && c == '\'') {
            flush();
            quote = true;
            continue;
        }
        if (quote) {
            if (c == '\'') quote = false;
            continue;
        }
        if (comment) continue;
        if (c == '%') {
            flush();
            comment = true;
            continue;
        }
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            word.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
        } else {
            flush();
        }
    }
    flush();
    return depth > 0;
}

}  // namespace cudamath
