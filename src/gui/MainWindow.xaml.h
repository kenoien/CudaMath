#pragma once

#include "MainWindow.g.h"

#include "pch.h"

#include <map>
#include <string>

namespace cudamath {
struct Session;
}

namespace winrt::CudaMathGui::implementation {

struct TabState {
    cudamath::Session* session{nullptr};
    Microsoft::UI::Xaml::Controls::TextBox output{nullptr};
    Microsoft::UI::Xaml::Controls::TextBox input{nullptr};
    Microsoft::UI::Xaml::Controls::ListView symbols{nullptr};
    Microsoft::UI::Xaml::Controls::TextBlock source{nullptr};
    Microsoft::UI::Xaml::Controls::TabViewItem item{nullptr};
};

struct MainWindow : MainWindowT<MainWindow> {
    MainWindow();

    void NewWorkspaceClick(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
    void OpenWorkspaceClick(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
    void SaveWorkspaceClick(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
    void OpenScriptClick(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
    void ExitClick(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
    void HelpClick(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
    void AboutClick(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
    void TimerClick(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
    void TabCloseRequested(Microsoft::UI::Xaml::Controls::TabView const&, Microsoft::UI::Xaml::Controls::TabViewTabCloseRequestedEventArgs const&);
    void SessionChanged(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const&);

private:
    int AddTab(cudamath::Session& session);
    int SelectedId();
    TabState* Find(int id);
    void Run(int id, const std::string& command, bool echo);
    void RefreshSymbols(int id);
    void RefreshHeader(int id);
    void RefreshStatus();
    void RemoveTab(int id);
    void Append(TabState& tab, const std::string& text);
    HWND WindowHandle();
    winrt::fire_and_forget CloseTabAsync(int id);
    winrt::fire_and_forget SaveAsync(int id, bool closeAfter);
    winrt::fire_and_forget OpenFileAsync(bool workspace);
    winrt::fire_and_forget ExitAsync();
    winrt::Windows::Foundation::IAsyncOperation<bool> PickAndSave(int id);
    void WindowClosing(Microsoft::UI::Windowing::AppWindow const&, Microsoft::UI::Windowing::AppWindowClosingEventArgs const&);

    std::map<int, TabState> tabs_;
    int nextId_{1};
    double cudaFreeBytes_{0};
    bool allowClose_{false};
    bool closing_{false};
    HWND hwnd_{nullptr};
    Microsoft::UI::Xaml::Window helpWindow_{nullptr};
};

}  // namespace winrt::CudaMathGui::implementation

namespace winrt::CudaMathGui::factory_implementation {

struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow> {};

}  // namespace winrt::CudaMathGui::factory_implementation
