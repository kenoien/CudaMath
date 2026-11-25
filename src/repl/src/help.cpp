#include "cudamath/repl.hpp"

#include <cctype>
#include <sstream>

namespace cudamath {

namespace {

const char* kTypes =
    "Types\n"
    "  double, single, int32, int64\n"
    "  A bare literal is double. Values are dense 2-D column-major arrays. A scalar is 1-by-1.\n"
    "  There is no logical type. A condition is true when it is nonempty and every element is nonzero.\n"
    "\n";

const char* kFunctions =
    "Functions\n"
    "  zeros, ones, eye, diag, linspace, repmat, reshape, tril, triu\n"
    "  size, numel, length, ndims, isempty, class, horzcat, vertcat, print, prn\n"
    "  double, single, int32, int64\n"
    "  rng, rand, randn, randi\n"
    "  abs, sign, sqrt, exp, log, log2, log10, sin, cos, tan\n"
    "  asin, acos, atan, atan2, sinh, cosh, tanh, floor, ceil, round, fix, mod, rem, power\n"
    "  sum, prod, mean, min, max, any, all, norm, dot, trace\n"
    "  inv, det, rank, lu, qr, chol, svd, eig, cross\n"
    "  Operators + - .* ./ .^ * / \\ ' .' and comparisons, &, |, ~\n"
    "  A user function takes its arguments in parentheses, separated by commas:\n"
    "    function y = add(a, b)\n"
    "      y = a + b\n"
    "    end\n"
    "  Call it as add(1, 2). The call passes one value for each name, in that order.\n"
    "  The function returns the value assigned to y.\n"
    "  function y = name() takes no arguments. function show(a) returns no value.\n"
    "  if/end, for/end, while/end\n"
    "  tests/end tests and test/end group expect checks\n"
    "\n"
    "Workspace files use .cmw. Scripts and suites use .cm.\n";

}  // namespace

std::string helpText() {
    return std::string(
        "Commands\n"
        "  help, ?              commands, types, and functions\n"
        "  who                  list defined names\n"
        "  whos                 name, type, and size\n"
        "  clear                clear one name, or the workspace\n"
        "  clear functions      remove user functions\n"
        "  saveworkspace path   write a .cmw workspace file\n"
        "  loadworkspace path   replace the workspace from a .cmw file\n"
        "  save(path, a, b)     write named variables to a .cma data file\n"
        "  load(path)           read a .cma data file into the workspace\n"
        "  run path             execute a .cm script in this session\n"
        "  clc                  clear the REPL output\n"
        "  expect               compare an actual value with an expected value\n"
        "  runtests path        run a .cm suite in a scratch workspace\n"
        "  timer on|off         report elapsed time after each submission\n"
        "  timer                print timer on or timer off\n"
        "  cudaversion          CUDA runtime version\n"
        "  memory               workspace bytes, CUDA total, and CUDA free\n"
        "  exit, quit           leave the console process\n"
        "\n") + kTypes + kFunctions;
}

std::string helpDocument() {
    return std::string(
        "App\n"
        "  CudaMath is a MATLAB-style REPL that calls the CUDA math libraries.\n"
        "  The console has one session. The window has one session per document window.\n"
        "  Each document window has a REPL and a list of that session's variables and user functions.\n"
        "  Selecting a function shows its source. Selecting a variable shows its elements.\n"
        "  Each workspace is a document window. Its title is the workspace name, and its status bar shows that window's timer and memory.\n"
        "  Window lists the open documents. Choosing one brings that window to the front.\n"
        "  A large command runs on a worker thread. The window stays responsive, the frame title says Running, and that document shows Running until the command finishes.\n"
        "  Another command waits its turn, and that document shows Waiting.\n"
        "  The frame status bar shows CUDA free memory and the CUDA version for the whole process.\n"
        "  Click the Timer button on a document's status bar to turn the timer on or off.\n"
        "  A tests/end tests file is a suite. Workspace files use .cmw. Scripts use .cm.\n"
        "\n") + kTypes + kFunctions;
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
    int nest = 0;
    std::string word;
    std::string previous;
    auto flush = [&]() {
        if (word.empty()) return;
        if (nest > 0) {
            previous.clear();
        } else if (word == "tests" && previous == "end") {
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
    char last = '\0';
    bool space = true;
    for (char c : input) {
        if (c == '\n') comment = false;
        if (!quote && c == '\'') {
            flush();
            const bool transpose = !space &&
                (std::isalnum(static_cast<unsigned char>(last)) || last == '_' || last == ')' || last == ']' || last == '.');
            last = '\'';
            space = false;
            if (!transpose) quote = true;
            continue;
        }
        if (quote) {
            if (c == '\'') quote = false;
            last = c;
            space = false;
            continue;
        }
        space = std::isspace(static_cast<unsigned char>(c)) != 0;
        if (!space) last = c;
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
            if (c == '(' || c == '[') ++nest;
            else if (nest > 0 && (c == ')' || c == ']')) --nest;
        }
    }
    flush();
    return depth > 0;
}

}  // namespace cudamath
