#include "pch.h"

#include "App.xaml.h"
#include "MainWindow.xaml.h"
#include "cudamath/engine.hpp"

#include <string>

namespace {

std::wstring widen(const std::string& text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
    if (!wide.empty() && wide.back() == L'\0') wide.pop_back();
    return wide;
}

}  // namespace

namespace winrt::CudaMathGui::implementation {

App::App() {
    InitializeComponent();
}

void App::OnLaunched(Microsoft::UI::Xaml::LaunchActivatedEventArgs const&) {
    std::string message;
    if (!cudamath::Engine::instance().startup(message)) {
        MessageBoxW(nullptr, widen(message).c_str(), L"CudaMath", MB_OK | MB_ICONERROR);
        ExitProcess(1);
    }
    window = make<MainWindow>();
    window.Activate();
}

}  // namespace winrt::CudaMathGui::implementation
