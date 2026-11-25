#pragma once

#include "cudamath/engine.hpp"

#include <string>
#include <string_view>

namespace cudamath {

struct ExecResult {
    std::string output;
    bool requestExit{false};
    bool clearOutput{false};
    bool showElapsed{false};
    double elapsedMs{0};
    bool incomplete{false};
};

// Evaluate one submission. Locks the engine so two sessions cannot overlap
// CUDA work. A submission whose control-flow blocks are still open returns
// incomplete and prints nothing.
ExecResult execute(Engine& engine, Session& session, std::string_view input);

// True when function/if/for/while/test/tests blocks are still open.
bool submissionIsOpen(std::string_view input);

std::string helpText();
std::string helpDocument();
std::string statusMemory(const Session& session);
std::string statusTimer(const Session& session);
std::string statusWorkspace(const Session& session);

}  // namespace cudamath
