#include "cudamath/engine.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <sstream>
#include <windows.h>

namespace cudamath {
namespace {

std::wstring utf8ToWide(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
    if (!wide.empty() && wide.back() == L'\0') {
        wide.pop_back();
    }
    return wide;
}

std::string failure(const std::string& detail) {
    std::string message = "CudaMath requires an NVIDIA GPU with CUDA unified memory.";
    if (!detail.empty()) {
        message += " ";
        message += detail;
    }
    return message;
}

}  // namespace

std::string cudaBinDirectory() {
    char path[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableA("CUDA_PATH", path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) {
        return {};
    }
    return std::string(path) + "\\bin\\arm64";
}

Engine& Engine::instance() {
    static Engine engine;
    return engine;
}

bool Engine::startup(std::string& message) {
    if (ready_) {
        return true;
    }
    const std::string bin = cudaBinDirectory();
    if (bin.empty()) {
        message = failure("CUDA_PATH is not set, so the CUDA runtime could not be loaded.");
        return false;
    }
    SetDllDirectoryW(utf8ToWide(bin).c_str());
    const std::wstring runtime = utf8ToWide(bin + "\\cudart64_13.dll");
    if (LoadLibraryW(runtime.c_str()) == nullptr) {
        message = failure("The CUDA runtime could not be loaded.");
        return false;
    }
    LoadLibraryW(utf8ToWide(bin + "\\cublas64_13.dll").c_str());
    LoadLibraryW(utf8ToWide(bin + "\\curand64_10.dll").c_str());
    LoadLibraryW(utf8ToWide(bin + "\\cusolver64_12.dll").c_str());

    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count < 1) {
        const char* text = status == cudaSuccess ? "No CUDA device is present." : cudaGetErrorString(status);
        message = failure(text);
        return false;
    }
    int concurrentDevice = -1;
    int managedDevice = -1;
    for (int device = 0; device < count; ++device) {
        int managed = 0;
        int concurrent = 0;
        cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device);
        cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device);
        if (managed != 0 && concurrent != 0 && concurrentDevice < 0) concurrentDevice = device;
        if (managed != 0 && managedDevice < 0) managedDevice = device;
    }
    const int chosen = concurrentDevice >= 0 ? concurrentDevice : managedDevice;
    if (chosen < 0) {
        message = failure("This GPU does not support CUDA unified memory.");
        return false;
    }
    status = cudaSetDevice(chosen);
    if (status != cudaSuccess) {
        message = failure(cudaGetErrorString(status));
        return false;
    }
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        message = failure("cuBLAS could not be created.");
        return false;
    }
    cublas_ = handle;
    cusolverDnHandle_t solver = nullptr;
    if (cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS) {
        message = failure("cuSOLVER could not be created.");
        return false;
    }
    solver_ = solver;
    int version = 0;
    cudaRuntimeGetVersion(&version);
    runtimeVersion_ = version;
    ready_ = true;
    return true;
}

std::string Engine::cudaVersion() const {
    const int major = runtimeVersion_ / 1000;
    const int minor = (runtimeVersion_ % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

MemoryInfo Engine::memory(const Session& session) const {
    MemoryInfo info;
    info.workspaceBytes = session.workspaceBytes();
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
        info.cudaFree = static_cast<double>(freeBytes);
        info.cudaTotal = static_cast<double>(totalBytes);
    }
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(0);
    out << "Workspace   " << info.workspaceBytes << " bytes\n"
        << "    Bytes stored in the variables defined right now. This is the sum of the arrays the engine allocated for this workspace.\n\n"
        << "CUDA total  " << info.cudaTotal << " bytes\n"
        << "    Size of the unified memory pool the CUDA driver reports. On this machine that pool is shared with Windows. It is not a separate graphics card.\n\n"
        << "CUDA free   " << info.cudaFree << " bytes\n"
        << "    Bytes in that pool the driver currently reports as free for new CUDA allocations. A managed array is counted here only once its pages are committed, so free memory can be larger than total minus the workspace.\n";
    info.text = out.str();
    return info;
}

namespace {

int unusedWorkspaceNumber(const std::vector<std::unique_ptr<Session>>& sessions) {
    int number = 1;
    for (;;) {
        bool used = false;
        for (const auto& session : sessions) {
            if (session->path.empty() && session->workspaceNumber == number) {
                used = true;
                break;
            }
        }
        if (!used) return number;
        ++number;
    }
}

}  // namespace

Session& Engine::createSession() {
    auto session = std::make_unique<Session>();
    session->workspaceNumber = unusedWorkspaceNumber(sessions_);
    sessions_.push_back(std::move(session));
    return *sessions_.back();
}

void Engine::destroySession(Session& session) {
    session.resetRng();
    for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
        if (it->get() == &session) {
            sessions_.erase(it);
            return;
        }
    }
}

std::vector<Session*> Engine::sessions() {
    std::vector<Session*> result;
    result.reserve(sessions_.size());
    for (auto& session : sessions_) {
        result.push_back(session.get());
    }
    return result;
}

}  // namespace cudamath
