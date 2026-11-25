#include "pch.h"

#include "MainWindow.xaml.h"

#include "cudamath/repl.hpp"

#include <microsoft.ui.xaml.window.h>
#include <shobjidl.h>

#include <fstream>
#include <sstream>

#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif

namespace winrt {
using namespace Microsoft::UI::Xaml;
using namespace Microsoft::UI::Xaml::Controls;
using namespace Microsoft::UI::Xaml::Input;
using namespace Microsoft::UI::Xaml::Media;
using namespace Microsoft::UI::Windowing;
using namespace Windows::Foundation;
using namespace Windows::Storage;
using namespace Windows::Storage::Pickers;
}  // namespace winrt

namespace {

std::wstring widen(const std::string& text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
    if (!wide.empty() && wide.back() == L'\0') wide.pop_back();
    return wide;
}

std::string narrow(winrt::hstring const& text) {
    if (text.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string utf8(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, utf8.data(), size, nullptr, nullptr);
    if (!utf8.empty() && utf8.back() == '\0') utf8.pop_back();
    return utf8;
}

std::string quotePath(const std::string& path) {
    if (path.find('\'') != std::string::npos) {
        throw std::runtime_error("paths cannot contain a single quote");
    }
    return "'" + path + "'";
}

std::wstring megabytes(double bytes) {
    std::wostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    out << (bytes / (1024.0 * 1024.0)) << L" MB";
    return out.str();
}

bool containsSuite(const std::string& source) {
    return source.find("tests") != std::string::npos && source.find("end tests") != std::string::npos;
}

}  // namespace

namespace winrt::CudaMathGui::implementation {

MainWindow::MainWindow() {
    InitializeComponent();
    Title(L"CudaMath");
    hwnd_ = WindowHandle();
    const UINT dpi = GetDpiForWindow(hwnd_);
    const float scale = static_cast<float>(dpi) / 96.0f;
    SetWindowPos(hwnd_, nullptr, 0, 0, static_cast<int>(1100 * scale), static_cast<int>(720 * scale), SWP_NOMOVE | SWP_NOZORDER);

    const auto windowId = Microsoft::UI::GetWindowIdFromWindow(hwnd_);
    auto appWindow = AppWindow::GetFromWindowId(windowId);
    appWindow.Closing({get_weak(), &MainWindow::WindowClosing});

    CudaStatus().Text(L"CUDA " + widen(cudamath::Engine::instance().cudaVersion()));
    AddTab(cudamath::Engine::instance().createSession());
}

HWND MainWindow::WindowHandle() {
    if (hwnd_ == nullptr) {
        Window window = *this;
        window.as<::IWindowNative>()->get_WindowHandle(&hwnd_);
    }
    return hwnd_;
}

int MainWindow::AddTab(cudamath::Session& session) {
    const int id = nextId_++;
    TabViewItem item;
    item.Header(box_value(widen(session.title())));
    item.Tag(box_value(id));
    item.HorizontalContentAlignment(HorizontalAlignment::Stretch);
    item.VerticalContentAlignment(VerticalAlignment::Stretch);

    Grid root;
    root.HorizontalAlignment(HorizontalAlignment::Stretch);
    root.VerticalAlignment(VerticalAlignment::Stretch);
    RowDefinition replRow;
    replRow.Height(GridLength{1, GridUnitType::Star});
    RowDefinition inputRow;
    inputRow.Height(GridLength{1, GridUnitType::Auto});
    root.RowDefinitions().Append(replRow);
    root.RowDefinitions().Append(inputRow);
    ColumnDefinition replColumn;
    replColumn.Width(GridLength{2, GridUnitType::Star});
    ColumnDefinition symbolColumn;
    symbolColumn.Width(GridLength{1, GridUnitType::Star});
    root.ColumnDefinitions().Append(replColumn);
    root.ColumnDefinitions().Append(symbolColumn);

    FontFamily consolas{L"Consolas"};
    TextBox output;
    output.IsReadOnly(true);
    output.AcceptsReturn(true);
    output.TextWrapping(TextWrapping::NoWrap);
    output.FontFamily(consolas);
    output.HorizontalAlignment(HorizontalAlignment::Stretch);
    output.VerticalAlignment(VerticalAlignment::Stretch);
    output.Margin(Thickness{8, 8, 4, 4});
    Grid::SetRow(output, 0);
    Grid::SetColumn(output, 0);

    TextBox input;
    input.AcceptsReturn(true);
    input.FontFamily(consolas);
    input.MinHeight(72);
    input.Margin(Thickness{8, 4, 4, 8});
    input.PlaceholderText(L"Enter runs the submission. Shift+Enter inserts a line.");
    Grid::SetRow(input, 1);
    Grid::SetColumn(input, 0);
    input.PreviewKeyDown([this, id](IInspectable const&, KeyRoutedEventArgs const& args) {
        if (args.Key() != Windows::System::VirtualKey::Enter) return;
        if ((GetKeyState(VK_SHIFT) & 0x8000) != 0) return;
        auto* tab = Find(id);
        if (tab == nullptr) return;
        const std::string text = narrow(tab->input.Text());
        if (cudamath::submissionIsOpen(text)) return;
        args.Handled(true);
        if (text.find_first_not_of(" \t\r\n") == std::string::npos) return;
        tab->input.Text(L"");
        Run(id, text, true);
    });

    ListView symbols;
    symbols.HorizontalAlignment(HorizontalAlignment::Stretch);
    symbols.VerticalAlignment(VerticalAlignment::Stretch);
    symbols.Margin(Thickness{4, 8, 8, 4});
    Grid::SetRow(symbols, 0);
    Grid::SetColumn(symbols, 1);
    symbols.SelectionChanged([this, id](IInspectable const&, SelectionChangedEventArgs const&) {
        auto* tab = Find(id);
        if (tab == nullptr || tab->symbols.SelectedItem() == nullptr) return;
        const std::string line = narrow(unbox_value<hstring>(tab->symbols.SelectedItem()));
        const auto name = line.substr(0, line.find(' '));
        const auto function = tab->session->functions.find(name);
        tab->source.Text(function == tab->session->functions.end() ? L"" : widen(function->second.body));
    });

    TextBlock source;
    source.FontFamily(consolas);
    source.TextWrapping(TextWrapping::Wrap);
    source.IsTextSelectionEnabled(true);
    source.Margin(Thickness{8, 4, 8, 8});
    Grid::SetRow(source, 1);
    Grid::SetColumn(source, 1);

    root.Children().Append(output);
    root.Children().Append(input);
    root.Children().Append(symbols);
    root.Children().Append(source);
    item.Content(root);
    Sessions().TabItems().Append(item);
    Sessions().SelectedItem(item);

    TabState state;
    state.session = &session;
    state.output = output;
    state.input = input;
    state.symbols = symbols;
    state.source = source;
    state.item = item;
    tabs_.emplace(id, std::move(state));
    RefreshSymbols(id);
    RefreshStatus();
    return id;
}

int MainWindow::SelectedId() {
    auto selected = Sessions().SelectedItem();
    if (!selected) return -1;
    auto item = selected.try_as<TabViewItem>();
    if (!item || !item.Tag()) return -1;
    return unbox_value<int>(item.Tag());
}

TabState* MainWindow::Find(int id) {
    auto it = tabs_.find(id);
    return it == tabs_.end() ? nullptr : &it->second;
}

void MainWindow::Append(TabState& tab, const std::string& text) {
    if (text.empty()) return;
    std::string next = narrow(tab.output.Text());
    next += text;
    if (next.back() != '\n') next.push_back('\n');
    tab.output.Text(widen(next));
}

void MainWindow::Run(int id, const std::string& command, bool echo) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    if (echo) Append(*tab, ">> " + command);
    const cudamath::ExecResult result = cudamath::execute(cudamath::Engine::instance(), *tab->session, command);
    if (result.clearOutput) tab->output.Text(L"");
    if (!result.output.empty()) Append(*tab, result.output);
    const auto memory = cudamath::Engine::instance().memory(*tab->session);
    cudaFreeBytes_ = memory.cudaFree;
    RefreshSymbols(id);
    RefreshHeader(id);
    RefreshStatus();
    if (result.requestExit) CloseTabAsync(id);
}

void MainWindow::RefreshSymbols(int id) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    tab->symbols.Items().Clear();
    for (const auto& entry : tab->session->vars) {
        tab->symbols.Items().Append(box_value(widen(entry.second.whosLine(entry.first))));
    }
    for (const auto& entry : tab->session->functions) {
        tab->symbols.Items().Append(box_value(widen(entry.first + "  function")));
    }
}

void MainWindow::RefreshHeader(int id) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    tab->item.Header(box_value(widen(tab->session->title())));
}

void MainWindow::RefreshStatus() {
    auto* tab = Find(SelectedId());
    if (tab == nullptr) {
        WorkspaceStatus().Text(L"");
        TimerStatus().Content(box_value(L"Timer off"));
        MemoryStatus().Text(L"Workspace used 0.0 MB    CUDA Free " + megabytes(cudaFreeBytes_));
        return;
    }
    WorkspaceStatus().Text(widen(cudamath::statusWorkspace(*tab->session)));
    TimerStatus().Content(box_value(widen(cudamath::statusTimer(*tab->session))));
    std::wstring memory = L"Workspace used " + megabytes(tab->session->workspaceBytes()) + L"    CUDA Free " + megabytes(cudaFreeBytes_);
    MemoryStatus().Text(memory);
}

void MainWindow::RemoveTab(int id) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    cudamath::Session* session = tab->session;
    TabViewItem item = tab->item;
    tabs_.erase(id);
    uint32_t index = 0;
    if (Sessions().TabItems().IndexOf(item, index)) {
        Sessions().TabItems().RemoveAt(index);
    }
    cudamath::Engine::instance().destroySession(*session);
    RefreshStatus();
}

void MainWindow::NewWorkspaceClick(IInspectable const&, RoutedEventArgs const&) {
    AddTab(cudamath::Engine::instance().createSession());
}

void MainWindow::OpenWorkspaceClick(IInspectable const&, RoutedEventArgs const&) {
    OpenFileAsync(true);
}

void MainWindow::SaveWorkspaceClick(IInspectable const&, RoutedEventArgs const&) {
    SaveAsync(SelectedId(), false);
}

void MainWindow::OpenScriptClick(IInspectable const&, RoutedEventArgs const&) {
    OpenFileAsync(false);
}

void MainWindow::ExitClick(IInspectable const&, RoutedEventArgs const&) {
    ExitAsync();
}

void MainWindow::HelpClick(IInspectable const&, RoutedEventArgs const&) {
    if (helpWindow_ == nullptr) {
        Window help;
        help.Title(L"CudaMath Help");
        TextBox box;
        box.IsReadOnly(true);
        box.AcceptsReturn(true);
        box.TextWrapping(TextWrapping::Wrap);
        box.FontFamily(FontFamily{L"Consolas"});
        box.Text(widen(cudamath::helpDocument()));
        help.Content(box);
        helpWindow_ = help;
    }
    helpWindow_.Activate();
}

void MainWindow::AboutClick(IInspectable const&, RoutedEventArgs const&) {
    ContentDialog dialog;
    dialog.XamlRoot(Content().XamlRoot());
    dialog.Title(box_value(L"CudaMath"));
    dialog.CloseButtonText(L"Close");
    std::wstring text =
        L"Version 0.1.0\n"
        L"Windows ARM64 for the NVIDIA N1X\n"
        L"CUDA toolkit 13.4\n"
        L"A MATLAB-style REPL that calls the CUDA math libraries.";
    dialog.Content(box_value(text));
    dialog.ShowAsync();
}

void MainWindow::TimerClick(IInspectable const&, RoutedEventArgs const&) {
    const int id = SelectedId();
    auto* tab = Find(id);
    if (tab == nullptr) return;
    Run(id, tab->session->timer ? "timer off" : "timer on", true);
}

void MainWindow::TabCloseRequested(TabView const&, TabViewTabCloseRequestedEventArgs const& args) {
    auto item = args.Tab();
    if (!item || !item.Tag()) return;
    CloseTabAsync(unbox_value<int>(item.Tag()));
}

void MainWindow::SessionChanged(IInspectable const&, SelectionChangedEventArgs const&) {
    RefreshStatus();
}

void MainWindow::WindowClosing(winrt::Microsoft::UI::Windowing::AppWindow const&, winrt::Microsoft::UI::Windowing::AppWindowClosingEventArgs const& args) {
    if (allowClose_) return;
    args.Cancel(true);
    if (!closing_) ExitAsync();
}

winrt::fire_and_forget MainWindow::CloseTabAsync(int id) {
    auto lifetime = get_strong();
    auto* tab = Find(id);
    if (tab == nullptr) co_return;
    if (tab->session->dirty) {
        ContentDialog dialog;
        dialog.XamlRoot(Content().XamlRoot());
        dialog.Title(box_value(L"CudaMath"));
        dialog.Content(box_value(L"Save changes to " + widen(tab->session->title()) + L"?"));
        dialog.PrimaryButtonText(L"Save");
        dialog.SecondaryButtonText(L"Don't save");
        dialog.CloseButtonText(L"Cancel");
        const auto result = co_await dialog.ShowAsync();
        if (result == ContentDialogResult::None) co_return;
        if (result == ContentDialogResult::Primary) {
            const bool saved = co_await PickAndSave(id);
            if (!saved) co_return;
        }
    }
    RemoveTab(id);
}

winrt::fire_and_forget MainWindow::SaveAsync(int id, bool closeAfter) {
    auto lifetime = get_strong();
    const bool saved = co_await PickAndSave(id);
    if (saved && closeAfter) RemoveTab(id);
}

winrt::Windows::Foundation::IAsyncOperation<bool> MainWindow::PickAndSave(int id) {
    auto lifetime = get_strong();
    auto* tab = Find(id);
    if (tab == nullptr) co_return false;
    std::string path = tab->session->path;
    if (path.empty()) {
        FileSavePicker picker;
        picker.as<::IInitializeWithWindow>()->Initialize(WindowHandle());
        picker.SuggestedStartLocation(PickerLocationId::DocumentsLibrary);
        picker.SuggestedFileName(L"workspace");
        picker.FileTypeChoices().Insert(L"CudaMath workspace", winrt::single_threaded_vector<hstring>({L".cmw"}));
        auto file = co_await picker.PickSaveFileAsync();
        if (!file) co_return false;
        path = narrow(file.Path());
        if (path.size() < 4 || path.substr(path.size() - 4) != ".cmw") path += ".cmw";
    }
    try {
        Run(id, "saveworkspace " + quotePath(path), false);
    } catch (const std::exception& error) {
        MessageBoxW(WindowHandle(), widen(error.what()).c_str(), L"CudaMath", MB_OK | MB_ICONERROR);
        co_return false;
    }
    co_return tab->session->path == path && !tab->session->dirty;
}

winrt::fire_and_forget MainWindow::OpenFileAsync(bool workspace) {
    auto lifetime = get_strong();
    FileOpenPicker picker;
    picker.as<::IInitializeWithWindow>()->Initialize(WindowHandle());
    picker.SuggestedStartLocation(PickerLocationId::DocumentsLibrary);
    picker.FileTypeFilter().Append(workspace ? L".cmw" : L".cm");
    auto file = co_await picker.PickSingleFileAsync();
    if (!file) co_return;
    const std::string path = narrow(file.Path());
    int id = SelectedId();
    if (workspace) {
        auto* active = Find(id);
        if (active == nullptr || !active->session->isPristine()) {
            id = AddTab(cudamath::Engine::instance().createSession());
        }
        Run(id, "loadworkspace " + quotePath(path), true);
        co_return;
    }
    if (id < 0) id = AddTab(cudamath::Engine::instance().createSession());
    std::string source;
    {
        std::ifstream stream(path, std::ios::binary);
        std::stringstream buffer;
        buffer << stream.rdbuf();
        source = buffer.str();
    }
    const std::string command = std::string(containsSuite(source) ? "runtests " : "run ") + quotePath(path);
    Run(id, command, true);
}

winrt::fire_and_forget MainWindow::ExitAsync() {
    auto lifetime = get_strong();
    if (closing_) co_return;
    closing_ = true;
    std::vector<int> ids;
    for (const auto& entry : tabs_) ids.push_back(entry.first);
    for (int id : ids) {
        auto* tab = Find(id);
        if (tab == nullptr || !tab->session->dirty) continue;
        ContentDialog dialog;
        dialog.XamlRoot(Content().XamlRoot());
        dialog.Title(box_value(L"CudaMath"));
        dialog.Content(box_value(L"Save changes to " + widen(tab->session->title()) + L"?"));
        dialog.PrimaryButtonText(L"Save");
        dialog.SecondaryButtonText(L"Don't save");
        dialog.CloseButtonText(L"Cancel");
        const auto result = co_await dialog.ShowAsync();
        if (result == ContentDialogResult::None) {
            closing_ = false;
            co_return;
        }
        if (result == ContentDialogResult::Primary) {
            const bool saved = co_await PickAndSave(id);
            if (!saved) {
                closing_ = false;
                co_return;
            }
        }
    }
    allowClose_ = true;
    Close();
}

}  // namespace winrt::CudaMathGui::implementation
