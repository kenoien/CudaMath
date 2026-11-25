#include "cudamath/repl.hpp"

#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int gPasses = 0;
int gFailures = 0;

void locateRepo() {
    wchar_t buffer[MAX_PATH]{};
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    std::filesystem::path dir = std::filesystem::path(buffer).parent_path();
    for (int step = 0; step < 8; ++step) {
        if (std::filesystem::exists(dir / "tests" / "alltests.cm")) {
            std::filesystem::current_path(dir);
            return;
        }
        if (!dir.has_parent_path() || dir == dir.parent_path()) break;
        dir = dir.parent_path();
    }
}

cudamath::Session& fresh(cudamath::Engine& engine) {
    return engine.createSession();
}

void check(bool ok, const std::string& name, const std::string& detail) {
    if (ok) {
        ++gPasses;
        std::cout << "PASS  " << name << '\n';
        return;
    }
    ++gFailures;
    std::cout << "FAIL  " << name;
    if (!detail.empty()) std::cout << ": " << detail;
    std::cout << '\n';
}

std::string run(cudamath::Engine& engine, cudamath::Session& session, const std::string& input) {
    return cudamath::execute(engine, session, input).output;
}

void checkCommand(cudamath::Engine& engine, const std::string& name, const std::string& input, const std::string& expected) {
    auto& session = fresh(engine);
    const auto output = run(engine, session, input);
    check(output == expected, name, output);
    engine.destroySession(session);
}

void checkWorkspaceCommands(cudamath::Engine& engine) {
    auto& session = fresh(engine);
    run(engine, session, "a = int32(4)");
    run(engine, session, "b = single(2)");
    const std::string who = run(engine, session, "who");
    check(who.find("a\n") != std::string::npos && who.find("b\n") != std::string::npos, "who lists names", who);
    const std::string whos = run(engine, session, "whos");
    check(whos.find("a  int32  1x1") != std::string::npos && whos.find("b  single  1x1") != std::string::npos, "whos lists type and size", whos);
    run(engine, session, "clear b");
    const auto missing = run(engine, session, "b");
    check(missing.rfind("Error:", 0) == 0, "clear name", missing);
    const auto kept = run(engine, session, "a");
    check(kept.rfind("Error:", 0) != 0, "clear name keeps the other variable", kept);
    run(engine, session, "function y = inc(x)\n  y = x + 1\nend");
    run(engine, session, "clear functions");
    const auto gone = run(engine, session, "inc(1)");
    check(gone.rfind("Error:", 0) == 0, "clear functions", gone);
    const auto still = run(engine, session, "a");
    check(still.rfind("Error:", 0) != 0, "clear functions keeps variables", still);
    run(engine, session, "clear");
    const auto cleared = run(engine, session, "a");
    check(cleared.rfind("Error:", 0) == 0, "clear workspace", cleared);
    engine.destroySession(session);
}

void checkTimerMemoryAndVersion(cudamath::Engine& engine) {
    auto& session = fresh(engine);
    check(run(engine, session, "timer") == "timer off\n", "timer", run(engine, session, "timer"));
    check(run(engine, session, "timer on") == "timer on\n", "timer on", "");
    check(session.timer, "timer on sets the session", "");
    const auto timed = cudamath::execute(engine, session, "1 + 1");
    check(timed.output.find("Elapsed ") != std::string::npos, "timer reports elapsed", timed.output);
    check(run(engine, session, "timer off") == "timer off\n" && !session.timer, "timer off", "");
    const auto version = cudamath::execute(engine, session, "cudaversion");
    const auto ans = session.vars.find("ans");
    check(version.output.find('.') != std::string::npos && ans != session.vars.end() && ans->second.isString &&
              ans->second.text + "\n" == version.output,
          "cudaversion", version.output);
    const auto memory = cudamath::execute(engine, session, "memory");
    const auto mem = session.vars.find("ans");
    check(memory.output.find("Workspace") != std::string::npos && memory.output.find("CUDA total") != std::string::npos &&
              memory.output.find("CUDA free") != std::string::npos && mem != session.vars.end() && mem->second.rows == 3 &&
              mem->second.cols == 1 && mem->second.type == cudamath::DType::F64,
          "memory", memory.output);
    const auto help = cudamath::execute(engine, session, "help");
    check(help.output == cudamath::helpText(), "help", help.output == cudamath::helpText() ? "" : "help text drifted");
    const auto cleared = cudamath::execute(engine, session, "clc");
    check(cleared.clearOutput && cleared.output.empty(), "clc", cleared.output);
    const auto leave = cudamath::execute(engine, session, "exit");
    check(leave.requestExit, "exit", "");
    const auto quit = cudamath::execute(engine, session, "quit");
    check(quit.requestExit, "quit", "");
    engine.destroySession(session);
}

void checkWorkspaceNames(cudamath::Engine& engine) {
    auto& first = engine.createSession();
    auto& second = engine.createSession();
    check(first.title() == "Workspace 1", "first unsaved workspace", first.title());
    check(second.title() == "Workspace 2", "second unsaved workspace", second.title());
    engine.destroySession(first);
    auto& reused = engine.createSession();
    check(reused.title() == "Workspace 1", "reused workspace number", reused.title());
    reused.dirty = true;
    check(reused.title() == "Workspace 1*", "dirty unsaved workspace", reused.title());
    second.path = "C:\\tmp\\notes.cmw";
    check(second.title() == "notes.cmw", "saved workspace uses the file name", second.title());
    auto& afterSave = engine.createSession();
    check(afterSave.title() == "Workspace 2", "saved workspace frees its number", afterSave.title());
    engine.destroySession(second);
    engine.destroySession(reused);
    engine.destroySession(afterSave);
}

void checkSuite(cudamath::Engine& engine, const std::string& path) {
    auto& session = fresh(engine);
    const auto result = cudamath::execute(engine, session, "runtests '" + path + "'");
    const auto ans = session.vars.find("ans");
    const bool passed = ans != session.vars.end() && !ans->second.isString && ans->second.get(0) == 0 &&
                        result.output.find("Error:") == std::string::npos;
    int suitePasses = 0;
    int suiteFails = 0;
    const auto marker = result.output.rfind(" passed, ");
    if (marker != std::string::npos) {
        const auto line = result.output.rfind('\n', marker);
        const auto start = line == std::string::npos ? 0 : line + 1;
        std::sscanf(result.output.c_str() + start, "%d passed, %d failed", &suitePasses, &suiteFails);
    }
    gPasses += suitePasses;
    gFailures += suiteFails;
    const std::string detail = std::to_string(suitePasses) + " passed, " + std::to_string(suiteFails) + " failed";
    if (passed && suiteFails == 0) std::cout << "PASS  " << path << ": " << detail << '\n';
    else {
        if (suitePasses == 0 && suiteFails == 0) ++gFailures;
        std::cout << "FAIL  " << path << ": " << detail << '\n' << result.output;
    }
    engine.destroySession(session);
}

}  // namespace

int main() {
    locateRepo();
    std::filesystem::create_directories("tests/_out");
    cudamath::Engine& engine = cudamath::Engine::instance();
    std::string message;
    if (!engine.startup(message)) {
        std::cerr << message << '\n';
        return 1;
    }
    checkCommand(engine, "help is a command", "help", cudamath::helpText());
    checkCommand(engine, "question mark is help", "?", cudamath::helpText());
    checkWorkspaceCommands(engine);
    checkTimerMemoryAndVersion(engine);
    checkWorkspaceNames(engine);
    {
        auto& session = fresh(engine);
        const auto summary = run(engine, session, "zeros(8, 9)");
        const auto listed = run(engine, session, "print(zeros(8, 9))");
        check(summary == "double 8x9\n", "large array is summarized", summary);
        check(listed.find('\n') != std::string::npos && listed.find("double") == std::string::npos && listed.rfind("0", 0) == 0,
              "print lists a large matrix", listed);
        engine.destroySession(session);
    }
    checkSuite(engine, "tests/alltests.cm");
    {
        std::ifstream file("tests/_out/big.cma", std::ios::binary);
        std::stringstream buffer;
        buffer << file.rdbuf();
        const std::string text = buffer.str();
        std::ifstream bin("tests/_out/big.cma.bin", std::ios::binary | std::ios::ate);
        const auto bytes = bin ? static_cast<std::int64_t>(bin.tellg()) : -1;
        const auto int32Bytes = 520ll * 520ll * 4ll;
        const auto fourBytes = 400ll * 350ll * 8ll + 560ll * 500ll * 4ll + int32Bytes + 380ll * 360ll * 8ll;
        std::ifstream eight("tests/_out/eight.cma", std::ios::binary);
        std::stringstream eightText;
        eightText << eight.rdbuf();
        std::ifstream int32bin("tests/_out/one-int32.cma.bin", std::ios::binary | std::ios::ate);
        std::ifstream fourbin("tests/_out/four.cma.bin", std::ios::binary | std::ios::ate);
        const auto int32Size = int32bin ? static_cast<std::int64_t>(int32bin.tellg()) : -1;
        const auto fourSize = fourbin ? static_cast<std::int64_t>(fourbin.tellg()) : -1;
        check(text.find("\"sidecar\"") != std::string::npos && bytes == 200ll * 700ll * 8ll,
              "large double uses a sidecar", text.substr(0, 180));
        check(int32Size == int32Bytes, "large int32 uses a sidecar", std::to_string(int32Size));
        check(fourSize == fourBytes, "four large matrices share one sidecar", std::to_string(fourSize));
        check(eightText.str().find("\"sidecar\"") == std::string::npos, "small and medium matrices stay inline",
              eightText.str().substr(0, 180));
        auto& session = fresh(engine);
        const auto missing = run(engine, session, "save('tests/_out/nope.cma', nosuch)");
        const auto wrongType = run(engine, session, "a = 1; save('tests/_out/nope.dat', a)");
        check(missing.find("undefined variable") != std::string::npos, "save rejects a missing name", missing);
        check(wrongType.find(".cma") != std::string::npos, "save rejects a non-cma path", wrongType);
        engine.destroySession(session);
    }
    std::cout << gPasses << " passed, " << gFailures << " failed\n";
    return gFailures == 0 ? 0 : 1;
}
