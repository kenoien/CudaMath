#pragma once

#include "cudamath/session.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cudamath {

struct MemoryInfo {
    double workspaceBytes{0};
    double cudaTotal{0};
    double cudaFree{0};
    std::string text;
};

// Process-wide CUDA context. Sessions share it. One submission runs at a time.
class Engine {
public:
    static Engine& instance();

    // Loads the CUDA runtime dynamically and checks unified memory.
    // On failure, message is the text both apps show, and no context is kept.
    bool startup(std::string& message);

    bool ready() const { return ready_; }
    std::string cudaVersion() const;
    MemoryInfo memory(const Session& session) const;

    Session& createSession();
    void destroySession(Session& session);
    std::vector<Session*> sessions();

    std::mutex& submissionMutex() { return submission_; }

    void* cublas() const { return cublas_; }
    void* solver() const { return solver_; }

private:
    Engine() = default;
    bool ready_{false};
    int runtimeVersion_{0};
    void* cublas_{nullptr};
    void* solver_{nullptr};
    std::mutex submission_;
    std::vector<std::unique_ptr<Session>> sessions_;
};

std::string cudaBinDirectory();

}  // namespace cudamath
