#include "cudamath/session.hpp"

#include <cuda_runtime.h>
#include <curand.h>

namespace cudamath {

double Session::workspaceBytes() const {
    double total = 0;
    for (const auto& entry : vars) {
        const Value& value = entry.second;
        if (!value.isString) {
            total += static_cast<double>(value.numel()) * static_cast<double>(dtypeSize(value.type));
        }
    }
    return total;
}

bool Session::isPristine() const {
    return path.empty() && vars.empty() && functions.empty() && !dirty;
}

std::string Session::title() const {
    std::string name;
    if (path.empty()) {
        name = "Workspace " + std::to_string(workspaceNumber > 0 ? workspaceNumber : 1);
    } else {
        name = path;
        const auto slash = name.find_last_of("\\/");
        if (slash != std::string::npos) name = name.substr(slash + 1);
    }
    if (dirty) {
        name.push_back('*');
    }
    return name;
}

void Session::resetRng() {
    if (rngAlive && rng != nullptr) {
        curandDestroyGenerator(reinterpret_cast<curandGenerator_t>(rng));
    }
    rng = nullptr;
    rngAlive = false;
}

}  // namespace cudamath
