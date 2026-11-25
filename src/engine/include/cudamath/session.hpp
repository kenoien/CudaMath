#pragma once

#include "cudamath/value.hpp"

#include <map>
#include <string>
#include <vector>

namespace cudamath {

struct UserFunction {
    std::string name;
    std::vector<std::string> outputs;
    std::vector<std::string> args;
    std::string body;
};

// One workspace. The console app has one. Each GUI document window has one.
// The cuRAND generator handle lives in the engine .cpp so this header
// stays free of CUDA types.
struct Session {
    std::string path;
    int workspaceNumber{0};
    bool dirty{false};
    bool timer{false};
    double lastElapsedMs{0};
    std::uint64_t rngSeed{0};
    std::uint64_t rngDrawn{0};
    bool rngAlive{false};
    void* rng{nullptr};

    std::map<std::string, Value> vars;
    std::map<std::string, UserFunction> functions;

    int suitePasses{0};
    int suiteFails{0};
    bool inSuite{false};
    std::string suiteName;
    std::string testName;

    double workspaceBytes() const;
    bool isPristine() const;
    std::string title() const;
    void resetRng();
};

}  // namespace cudamath
