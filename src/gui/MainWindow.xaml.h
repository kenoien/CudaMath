#pragma once

#include "MainWindow.g.h"

#include "pch.h"

#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
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
    Microsoft::UI::Xaml::Controls::Border frame{nullptr};
    Microsoft::UI::Xaml::Controls::Border titleBar{nullptr};
    Microsoft::UI::Xaml::Controls::TextBlock title{nullptr};
    Microsoft::UI::Xaml::Controls::Button timer{nullptr};
    Microsoft::UI::Xaml::Controls::TextBlock activity{nullptr};
    Microsoft::UI::Xaml::Controls::TextBlock memory{nullptr};
    Microsoft::UI::Xaml::Controls::TextBlock hint{nullptr};
    bool maximized{false};
    double restoreLeft{0};
    double restoreTop{0};
    double restoreWidth{0};
    double restoreHeight{0};
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

private:
    int AddTab(cudamath::Session& session);
    void Activate(int id);
    void AttachResize(Microsoft::UI::Xaml::Controls::Border const& strip, int id, int edge);
    void ApplyResize(Microsoft::UI::Xaml::Input::PointerRoutedEventArgs const& args);
    void ToggleMaximize(int id);
    int SelectedId();
    TabState* Find(int id);
    struct Job {
        int id{0};
        cudamath::Session* session{nullptr};
        std::string command;
        std::function<void()> done;
    };

    void Run(int id, const std::string& command, bool echo);
    winrt::Windows::Foundation::IAsyncAction RunAsync(int id, std::string command, bool echo);
    winrt::Windows::Foundation::IAsyncAction WaitUntilIdle();
    void Enqueue(int id, cudamath::Session* session, std::string command, std::function<void()> done);
    void StartWorker();
    void OnUi(std::function<void()> work);
    void UpdateActivity();
    bool Busy();
    bool DocumentBusy(int id);
    bool Queued(int id);
    void DropQueued(int id);
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
    int activeId_{-1};
    int zOrder_{1};
    int gestureId_{-1};
    bool dragging_{false};
    bool resizing_{false};
    int resizeEdge_{0};
    double gestureX_{0};
    double gestureY_{0};
    double resizeLeft_{0};
    double resizeTop_{0};
    double resizeWidth_{0};
    double resizeHeight_{0};
    double cudaFreeBytes_{0};
    std::mutex jobsMutex_;
    std::deque<Job> jobs_;
    bool workerRunning_{false};
    int runningId_{-1};
    std::set<int> pendingClose_;
    Microsoft::UI::Dispatching::DispatcherQueue uiQueue_{nullptr};
    bool allowClose_{false};
    bool closing_{false};
    HWND hwnd_{nullptr};
    Microsoft::UI::Xaml::Window helpWindow_{nullptr};
};

}  // namespace winrt::CudaMathGui::implementation

namespace winrt::CudaMathGui::factory_implementation {

struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow> {};

}  // namespace winrt::CudaMathGui::factory_implementation
