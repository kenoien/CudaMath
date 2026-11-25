#include "cudamath/repl.hpp"

#include <iostream>
#include <string>

namespace {

int runSubmission(cudamath::Engine& engine, cudamath::Session& session, const std::string& input) {
    const cudamath::ExecResult result = cudamath::execute(engine, session, input);
    if (result.clearOutput) {
        std::cout << "\x1b[2J\x1b[H";
    }
    if (!result.output.empty()) {
        std::cout << result.output;
        if (result.output.back() != '\n') std::cout << '\n';
    }
    return result.requestExit ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    cudamath::Engine& engine = cudamath::Engine::instance();
    std::string message;
    if (!engine.startup(message)) {
        std::cerr << message << '\n';
        return 1;
    }
    cudamath::Session& session = engine.createSession();
    if (argc >= 3 && std::string(argv[1]) == "--runtests") {
        const std::string command = std::string("runtests '") + argv[2] + "'";
        const cudamath::ExecResult result = cudamath::execute(engine, session, command);
        if (!result.output.empty()) std::cout << result.output;
        const auto ans = session.vars.find("ans");
        if (ans == session.vars.end() || ans->second.get(0) != 0) return 1;
        return 0;
    }
    std::string pending;
    while (true) {
        std::cout << (pending.empty() ? ">> " : ".. ") << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) break;
        if (!pending.empty()) pending.push_back('\n');
        pending += line;
        if (cudamath::submissionIsOpen(pending)) continue;
        const int code = runSubmission(engine, session, pending);
        pending.clear();
        if (code != 0) return 0;
    }
    return 0;
}
