#include "pch.h"

#include "MainWindow.xaml.h"

#include <winrt/Microsoft.UI.Input.h>
#include <winrt/Windows.UI.Text.h>

#include "cudamath/repl.hpp"

#include <microsoft.ui.xaml.window.h>
#include <shobjidl.h>

#include <coroutine>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <sstream>
#include <thread>

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

std::string symbolName(const std::string& line) {
    const auto end = line.find(' ');
    return line.substr(0, end);
}

std::string numberText(double number) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.15g", number);
    return buffer;
}

std::string matrixText(const cudamath::Value& value) {
    if (value.isString) return value.text;
    std::ostringstream out;
    out << cudamath::dtypeName(value.type) << ' ' << value.rows << 'x' << value.cols;
    if (value.empty()) {
        out << "\n[]";
        return out.str();
    }
    const bool full = value.rows <= 24 && value.cols <= 12;
    const auto shownRows = full ? value.rows : std::min<std::int64_t>(value.rows, 24);
    const auto shownCols = full ? value.cols : std::min<std::int64_t>(value.cols, 12);
    out << '\n';
    for (std::int64_t row = 0; row < shownRows; ++row) {
        if (row > 0) out << '\n';
        for (std::int64_t col = 0; col < shownCols; ++col) {
            if (col > 0) out << ' ';
            out << numberText(value.get(col * value.rows + row));
        }
        if (value.cols > shownCols) out << " ...";
    }
    if (value.rows > shownRows) out << "\n...";
    return out.str();
}

std::string symbolText(const cudamath::Session& session, const std::string& name) {
    const auto function = session.functions.find(name);
    if (function != session.functions.end()) return function->second.body;
    const auto variable = session.vars.find(name);
    if (variable == session.vars.end()) return {};
    return matrixText(variable->second);
}

double placed(double value) {
    return std::isfinite(value) ? value : 0;
}

winrt::Brush themeBrush(wchar_t const* key) {
    return winrt::Application::Current().Resources().Lookup(winrt::box_value(winrt::hstring(key))).as<winrt::Brush>();
}

struct UiResume {
    winrt::Microsoft::UI::Dispatching::DispatcherQueue queue{nullptr};
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<> handle) const {
        if (!queue.TryEnqueue([handle] { handle.resume(); })) handle.resume();
    }
    void await_resume() const noexcept {}
};

}  // namespace

namespace winrt::CudaMathGui::implementation {

MainWindow::MainWindow() {
    InitializeComponent();
    uiQueue_ = DispatcherQueue();
    Title(L"CudaMath");
    hwnd_ = WindowHandle();
    const UINT dpi = GetDpiForWindow(hwnd_);
    const float scale = static_cast<float>(dpi) / 96.0f;
    SetWindowPos(hwnd_, nullptr, 0, 0, static_cast<int>(1100 * scale), static_cast<int>(720 * scale), SWP_NOMOVE | SWP_NOZORDER);

    const auto windowId = Microsoft::UI::GetWindowIdFromWindow(hwnd_);
    auto appWindow = AppWindow::GetFromWindowId(windowId);
    appWindow.Closing({get_weak(), &MainWindow::WindowClosing});

    auto& engine = cudamath::Engine::instance();
    CudaStatus().Text(L"CUDA " + widen(engine.cudaVersion()));
    auto& session = engine.createSession();
    cudaFreeBytes_ = engine.memory(session).cudaFree;
    MdiClient().SizeChanged([this](IInspectable const&, SizeChangedEventArgs const& args) {
        for (auto& entry : tabs_) {
            if (!entry.second.maximized) continue;
            entry.second.frame.Width(args.NewSize().Width);
            entry.second.frame.Height(args.NewSize().Height);
        }
    });
    AddTab(session);
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
    output.Margin(Thickness{6, 6, 3, 3});
    Grid::SetRow(output, 0);
    Grid::SetColumn(output, 0);

    TextBox input;
    input.AcceptsReturn(true);
    input.FontFamily(consolas);
    input.MinHeight(72);
    input.Padding(Thickness{8, 6, 6, 6});
    Grid inputHost;
    inputHost.Margin(Thickness{6, 3, 3, 6});
    TextBlock hint;
    hint.Text(L"Enter expression here");
    hint.FontFamily(consolas);
    hint.FontStyle(Windows::UI::Text::FontStyle::Italic);
    hint.Foreground(SolidColorBrush{Windows::UI::Color{255, 128, 128, 128}});
    hint.IsHitTestVisible(false);
    hint.VerticalAlignment(VerticalAlignment::Top);
    hint.HorizontalAlignment(HorizontalAlignment::Left);
    hint.Margin(Thickness{9, 7, 8, 4});
    input.TextChanged([this, id, hint](IInspectable const& sender, TextChangedEventArgs const&) {
        const bool empty = sender.as<TextBox>().Text().empty();
        hint.Visibility(empty ? Visibility::Visible : Visibility::Collapsed);
        if (!empty) return;
        const bool running = id == runningId_;
        hint.Text(running ? L"Running\u2026" : L"Enter expression here");
        hint.FontStyle(running ? winrt::Windows::UI::Text::FontStyle::Normal : winrt::Windows::UI::Text::FontStyle::Italic);
    });
    inputHost.Children().Append(input);
    inputHost.Children().Append(hint);
    Grid::SetRow(inputHost, 1);
    Grid::SetColumn(inputHost, 0);
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
    symbols.FontFamily(consolas);
    symbols.HorizontalAlignment(HorizontalAlignment::Stretch);
    symbols.VerticalAlignment(VerticalAlignment::Stretch);
    symbols.Margin(Thickness{3, 6, 6, 3});
    Grid::SetRow(symbols, 0);
    Grid::SetColumn(symbols, 1);
    symbols.SelectionChanged([this, id](IInspectable const&, SelectionChangedEventArgs const&) {
        auto* tab = Find(id);
        if (tab == nullptr || tab->symbols.SelectedItem() == nullptr || id == runningId_) return;
        const std::string name = symbolName(narrow(unbox_value<hstring>(tab->symbols.SelectedItem())));
        tab->source.Text(widen(symbolText(*tab->session, name)));
    });

    TextBlock source;
    source.FontFamily(consolas);
    source.TextWrapping(TextWrapping::NoWrap);
    source.IsTextSelectionEnabled(true);
    ScrollViewer detail;
    detail.Content(source);
    detail.MaxHeight(240);
    detail.HorizontalScrollBarVisibility(ScrollBarVisibility::Auto);
    detail.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);
    detail.Margin(Thickness{6, 3, 6, 6});
    Grid::SetRow(detail, 1);
    Grid::SetColumn(detail, 1);

    root.Children().Append(output);
    root.Children().Append(inputHost);
    root.Children().Append(symbols);
    root.Children().Append(detail);

    TextBlock title;
    title.Text(widen(session.title()));
    title.VerticalAlignment(VerticalAlignment::Center);
    title.Margin(Thickness{10, 0, 8, 0});
    title.Foreground(themeBrush(L"TextOnAccentFillColorPrimaryBrush"));

    Button close;
    close.Content(box_value(L"\u2715"));
    close.Background(SolidColorBrush{Windows::UI::Colors::Transparent()});
    close.BorderThickness(Thickness{0});
    close.Padding(Thickness{10, 0, 10, 0});
    close.VerticalAlignment(VerticalAlignment::Stretch);
    close.Click([this, id](IInspectable const&, RoutedEventArgs const&) { CloseTabAsync(id); });

    Grid titleGrid;
    ColumnDefinition titleColumn;
    titleColumn.Width(GridLength{1, GridUnitType::Star});
    ColumnDefinition closeColumn;
    closeColumn.Width(GridLength{0, GridUnitType::Auto});
    titleGrid.ColumnDefinitions().Append(titleColumn);
    titleGrid.ColumnDefinitions().Append(closeColumn);
    titleGrid.Children().Append(title);
    Grid::SetColumn(close, 1);
    titleGrid.Children().Append(close);

    Border titleBar;
    titleBar.Height(32);
    titleBar.Child(titleGrid);
    titleBar.Background(themeBrush(L"AccentFillColorDefaultBrush"));
    titleBar.PointerPressed([this, id, close](IInspectable const& sender, PointerRoutedEventArgs const& args) {
        DependencyObject node = args.OriginalSource().try_as<DependencyObject>();
        const auto closeNode = close.as<DependencyObject>();
        while (node) {
            if (node == closeNode) return;
            node = VisualTreeHelper::GetParent(node);
        }
        Activate(id);
        auto* tab = Find(id);
        if (tab == nullptr || tab->maximized) return;
        dragging_ = true;
        resizing_ = false;
        gestureId_ = id;
        const auto point = args.GetCurrentPoint(MdiClient()).Position();
        gestureX_ = point.X - placed(Canvas::GetLeft(tab->frame));
        gestureY_ = point.Y - placed(Canvas::GetTop(tab->frame));
        sender.as<Border>().CapturePointer(args.Pointer());
        args.Handled(true);
    });
    titleBar.PointerMoved([this](IInspectable const&, PointerRoutedEventArgs const& args) {
        if (!dragging_ || gestureId_ < 0) return;
        auto* tab = Find(gestureId_);
        if (tab == nullptr) return;
        const auto point = args.GetCurrentPoint(MdiClient()).Position();
        Canvas::SetLeft(tab->frame, point.X - gestureX_);
        Canvas::SetTop(tab->frame, point.Y - gestureY_);
    });
    titleBar.PointerReleased([this](IInspectable const& sender, PointerRoutedEventArgs const& args) {
        dragging_ = false;
        gestureId_ = -1;
        sender.as<Border>().ReleasePointerCapture(args.Pointer());
    });
    titleBar.DoubleTapped([this, id](IInspectable const&, DoubleTappedRoutedEventArgs const&) { ToggleMaximize(id); });

    Button timer;
    timer.Height(22);
    timer.MinHeight(22);
    timer.MinWidth(0);
    timer.Padding(Thickness{10, 0, 10, 0});
    timer.Margin(Thickness{8, 0, 0, 0});
    timer.VerticalAlignment(VerticalAlignment::Center);
    timer.HorizontalAlignment(HorizontalAlignment::Left);
    timer.Background(themeBrush(L"ControlFillColorDefaultBrush"));
    timer.BorderBrush(themeBrush(L"ControlStrokeColorDefaultBrush"));
    timer.BorderThickness(Thickness{1});
    timer.CornerRadius(CornerRadius{4});
    timer.Foreground(themeBrush(L"TextFillColorPrimaryBrush"));
    ToolTip timerTip;
    timerTip.Content(box_value(L"Turn the timer on or off"));
    ToolTipService::SetToolTip(timer, timerTip);
    timer.Click([this, id](IInspectable const&, RoutedEventArgs const&) {
        Activate(id);
        auto* tab = Find(id);
        if (tab == nullptr) return;
        Run(id, tab->session->timer ? "timer off" : "timer on", true);
    });

    TextBlock activity;
    activity.VerticalAlignment(VerticalAlignment::Center);
    activity.Margin(Thickness{12, 0, 4, 0});
    activity.FontWeight(winrt::Windows::UI::Text::FontWeights::SemiBold());
    activity.Visibility(Visibility::Collapsed);

    TextBlock memoryCaption;
    memoryCaption.Text(L"Memory");
    memoryCaption.VerticalAlignment(VerticalAlignment::Center);
    memoryCaption.Foreground(themeBrush(L"TextFillColorSecondaryBrush"));
    TextBlock memory;
    memory.Text(L"0.0 MB");
    memory.VerticalAlignment(VerticalAlignment::Center);
    StackPanel memoryCell;
    memoryCell.Orientation(Orientation::Horizontal);
    memoryCell.Spacing(8);
    memoryCell.VerticalAlignment(VerticalAlignment::Center);
    memoryCell.Children().Append(memoryCaption);
    memoryCell.Children().Append(memory);

    Border memoryHost;
    memoryHost.BorderBrush(themeBrush(L"DividerStrokeColorDefaultBrush"));
    memoryHost.BorderThickness(Thickness{1, 0, 0, 0});
    memoryHost.Padding(Thickness{12, 0, 12, 0});
    memoryHost.Child(memoryCell);

    Grid childStatus;
    ColumnDefinition timerColumn;
    timerColumn.Width(GridLength{0, GridUnitType::Auto});
    ColumnDefinition activityColumn;
    activityColumn.Width(GridLength{0, GridUnitType::Auto});
    ColumnDefinition memoryColumn;
    memoryColumn.Width(GridLength{0, GridUnitType::Auto});
    ColumnDefinition restColumn;
    restColumn.Width(GridLength{1, GridUnitType::Star});
    childStatus.ColumnDefinitions().Append(timerColumn);
    childStatus.ColumnDefinitions().Append(activityColumn);
    childStatus.ColumnDefinitions().Append(memoryColumn);
    childStatus.ColumnDefinitions().Append(restColumn);
    childStatus.Children().Append(timer);
    Grid::SetColumn(activity, 1);
    childStatus.Children().Append(activity);
    Grid::SetColumn(memoryHost, 2);
    childStatus.Children().Append(memoryHost);

    Border childStatusBar;
    childStatusBar.Height(28);
    childStatusBar.BorderBrush(themeBrush(L"DividerStrokeColorDefaultBrush"));
    childStatusBar.BorderThickness(Thickness{0, 1, 0, 0});
    childStatusBar.Background(themeBrush(L"LayerFillColorDefaultBrush"));
    childStatusBar.Child(childStatus);

    Grid shell;
    RowDefinition titleRow;
    titleRow.Height(GridLength{0, GridUnitType::Auto});
    RowDefinition bodyRow;
    bodyRow.Height(GridLength{1, GridUnitType::Star});
    RowDefinition statusRow;
    statusRow.Height(GridLength{0, GridUnitType::Auto});
    shell.RowDefinitions().Append(titleRow);
    shell.RowDefinitions().Append(bodyRow);
    shell.RowDefinitions().Append(statusRow);
    shell.Children().Append(titleBar);
    Grid::SetRow(root, 1);
    shell.Children().Append(root);
    Grid::SetRow(childStatusBar, 2);
    shell.Children().Append(childStatusBar);
    shell.Background(themeBrush(L"SolidBackgroundFillColorBaseBrush"));

    const auto hit = SolidColorBrush{Windows::UI::Colors::Transparent()};
    auto addEdge = [&](HorizontalAlignment horizontal, VerticalAlignment vertical, double width, double height, int edge, winrt::Microsoft::UI::Input::InputSystemCursorShape cursor) {
        Border strip;
        strip.Background(hit);
        strip.HorizontalAlignment(horizontal);
        strip.VerticalAlignment(vertical);
        if (width > 0) strip.Width(width);
        if (height > 0) strip.Height(height);
        strip.as<IUIElementProtected>().ProtectedCursor(winrt::Microsoft::UI::Input::InputSystemCursor::Create(cursor));
        Grid::SetRowSpan(strip, 3);
        shell.Children().Append(strip);
        AttachResize(strip, id, edge);
    };
    using Cursor = winrt::Microsoft::UI::Input::InputSystemCursorShape;
    addEdge(HorizontalAlignment::Left, VerticalAlignment::Stretch, 6, 0, 1, Cursor::SizeWestEast);
    addEdge(HorizontalAlignment::Right, VerticalAlignment::Stretch, 6, 0, 2, Cursor::SizeWestEast);
    addEdge(HorizontalAlignment::Stretch, VerticalAlignment::Top, 0, 6, 4, Cursor::SizeNorthSouth);
    addEdge(HorizontalAlignment::Stretch, VerticalAlignment::Bottom, 0, 6, 8, Cursor::SizeNorthSouth);
    addEdge(HorizontalAlignment::Left, VerticalAlignment::Top, 12, 12, 5, Cursor::SizeNorthwestSoutheast);
    addEdge(HorizontalAlignment::Right, VerticalAlignment::Top, 12, 12, 6, Cursor::SizeNortheastSouthwest);
    addEdge(HorizontalAlignment::Left, VerticalAlignment::Bottom, 12, 12, 9, Cursor::SizeNortheastSouthwest);
    addEdge(HorizontalAlignment::Right, VerticalAlignment::Bottom, 12, 12, 10, Cursor::SizeNorthwestSoutheast);

    Border frame;
    frame.Child(shell);
    frame.Padding(Thickness{1});
    frame.Background(themeBrush(L"ControlStrongStrokeColorDefaultBrush"));
    frame.PointerPressed([this, id](IInspectable const&, PointerRoutedEventArgs const&) { Activate(id); });
    const double offset = 28.0 * static_cast<double>((id - 1) % 8);
    Canvas::SetLeft(frame, 16 + offset);
    Canvas::SetTop(frame, 16 + offset);
    frame.Width(760);
    frame.Height(480);
    frame.Loaded([this, id](IInspectable const&, RoutedEventArgs const&) {
        auto* tab = Find(id);
        if (tab == nullptr || tab->maximized) return;
        const double availW = MdiClient().ActualWidth();
        const double availH = MdiClient().ActualHeight();
        if (availW < 100 || availH < 100) return;
        const double left = placed(Canvas::GetLeft(tab->frame));
        const double top = placed(Canvas::GetTop(tab->frame));
        const double maxW = (std::max)(420.0, availW - left - 16);
        const double maxH = (std::max)(280.0, availH - top - 16);
        if (tab->frame.Width() > maxW) tab->frame.Width(maxW);
        if (tab->frame.Height() > maxH) tab->frame.Height(maxH);
    });
    MdiClient().Children().Append(frame);

    TabState state;
    state.session = &session;
    state.output = output;
    state.input = input;
    state.symbols = symbols;
    state.source = source;
    state.frame = frame;
    state.titleBar = titleBar;
    state.title = title;
    state.timer = timer;
    state.activity = activity;
    state.memory = memory;
    state.hint = hint;
    tabs_.emplace(id, std::move(state));
    Activate(id);
    RefreshSymbols(id);
    RefreshStatus();
    return id;
}

void MainWindow::AttachResize(Border const& strip, int id, int edge) {
    strip.PointerPressed([this, id, edge](IInspectable const& sender, PointerRoutedEventArgs const& args) {
        Activate(id);
        auto* tab = Find(id);
        if (tab == nullptr || tab->maximized) return;
        resizing_ = true;
        dragging_ = false;
        gestureId_ = id;
        resizeEdge_ = edge;
        const auto point = args.GetCurrentPoint(MdiClient()).Position();
        gestureX_ = point.X;
        gestureY_ = point.Y;
        resizeLeft_ = placed(Canvas::GetLeft(tab->frame));
        resizeTop_ = placed(Canvas::GetTop(tab->frame));
        resizeWidth_ = tab->frame.ActualWidth() > 0 ? tab->frame.ActualWidth() : tab->frame.Width();
        resizeHeight_ = tab->frame.ActualHeight() > 0 ? tab->frame.ActualHeight() : tab->frame.Height();
        sender.as<Border>().CapturePointer(args.Pointer());
        args.Handled(true);
    });
    strip.PointerMoved([this](IInspectable const&, PointerRoutedEventArgs const& args) { ApplyResize(args); });
    strip.PointerReleased([this](IInspectable const& sender, PointerRoutedEventArgs const& args) {
        resizing_ = false;
        gestureId_ = -1;
        sender.as<Border>().ReleasePointerCapture(args.Pointer());
    });
    strip.PointerCaptureLost([this](IInspectable const&, PointerRoutedEventArgs const&) {
        resizing_ = false;
        gestureId_ = -1;
    });
}

void MainWindow::ApplyResize(PointerRoutedEventArgs const& args) {
    if (!resizing_ || gestureId_ < 0) return;
    auto* tab = Find(gestureId_);
    if (tab == nullptr) return;
    const auto point = args.GetCurrentPoint(MdiClient()).Position();
    const double dx = point.X - gestureX_;
    const double dy = point.Y - gestureY_;
    double left = resizeLeft_;
    double top = resizeTop_;
    double width = resizeWidth_;
    double height = resizeHeight_;
    if ((resizeEdge_ & 2) != 0) width = (std::max)(420.0, resizeWidth_ + dx);
    if ((resizeEdge_ & 8) != 0) height = (std::max)(280.0, resizeHeight_ + dy);
    if ((resizeEdge_ & 1) != 0) {
        width = (std::max)(420.0, resizeWidth_ - dx);
        left = resizeLeft_ + (resizeWidth_ - width);
    }
    if ((resizeEdge_ & 4) != 0) {
        height = (std::max)(280.0, resizeHeight_ - dy);
        top = resizeTop_ + (resizeHeight_ - height);
    }
    Canvas::SetLeft(tab->frame, left);
    Canvas::SetTop(tab->frame, top);
    tab->frame.Width(width);
    tab->frame.Height(height);
}

void MainWindow::Activate(int id) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    activeId_ = id;
    Canvas::SetZIndex(tab->frame, ++zOrder_);
    RefreshStatus();
}

void MainWindow::ToggleMaximize(int id) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    Activate(id);
    if (!tab->maximized) {
        tab->restoreLeft = placed(Canvas::GetLeft(tab->frame));
        tab->restoreTop = placed(Canvas::GetTop(tab->frame));
        tab->restoreWidth = tab->frame.Width();
        tab->restoreHeight = tab->frame.Height();
        tab->maximized = true;
        Canvas::SetLeft(tab->frame, 0);
        Canvas::SetTop(tab->frame, 0);
        const double width = MdiClient().ActualWidth();
        const double height = MdiClient().ActualHeight();
        if (width > 0) tab->frame.Width(width);
        if (height > 0) tab->frame.Height(height);
        return;
    }
    tab->maximized = false;
    Canvas::SetLeft(tab->frame, tab->restoreLeft);
    Canvas::SetTop(tab->frame, tab->restoreTop);
    tab->frame.Width(tab->restoreWidth);
    tab->frame.Height(tab->restoreHeight);
}

int MainWindow::SelectedId() {
    return activeId_;
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
    if (closing_) return;
    auto* tab = Find(id);
    if (tab == nullptr) return;
    if (echo) Append(*tab, ">> " + command);
    Enqueue(id, tab->session, command, nullptr);
    UpdateActivity();
}

winrt::Windows::Foundation::IAsyncAction MainWindow::RunAsync(int id, std::string command, bool echo) {
    auto lifetime = get_strong();
    auto* tab = Find(id);
    if (tab == nullptr) co_return;
    if (echo) Append(*tab, ">> " + command);
    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    Enqueue(id, tab->session, std::move(command), [done] {
        try {
            done->set_value();
        } catch (...) {
        }
    });
    UpdateActivity();
    co_await winrt::resume_background();
    future.wait();
    co_await UiResume{uiQueue_};
}

winrt::Windows::Foundation::IAsyncAction MainWindow::WaitUntilIdle() {
    auto lifetime = get_strong();
    while (Busy()) {
        co_await winrt::resume_after(std::chrono::milliseconds(50));
        co_await UiResume{uiQueue_};
    }
}

void MainWindow::Enqueue(int id, cudamath::Session* session, std::string command, std::function<void()> done) {
    bool start = false;
    {
        std::lock_guard<std::mutex> lock(jobsMutex_);
        if (!workerRunning_) {
            workerRunning_ = true;
            start = true;
        }
        jobs_.push_back(Job{id, session, std::move(command), std::move(done)});
    }
    if (start) {
        runningId_ = id;
        StartWorker();
    }
}

void MainWindow::StartWorker() {
    auto lifetime = get_strong();
    std::thread([this, lifetime] {
        cudamath::Engine::instance().bindThread();
        for (;;) {
            Job job;
            {
                std::lock_guard<std::mutex> lock(jobsMutex_);
                if (jobs_.empty()) {
                    workerRunning_ = false;
                    return;
                }
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            auto work = std::make_shared<Job>(std::move(job));
            OnUi([this, id = work->id] {
                runningId_ = id;
                UpdateActivity();
            });
            cudamath::ExecResult result;
            double freeBytes = cudaFreeBytes_;
            try {
                result = cudamath::execute(cudamath::Engine::instance(), *work->session, work->command);
                freeBytes = cudamath::Engine::instance().memory(*work->session).cudaFree;
            } catch (const std::exception& error) {
                result.output = std::string("Error: ") + error.what() + "\n";
            }
            OnUi([this, work, result = std::move(result), freeBytes] {
                auto* tab = Find(work->id);
                if (tab != nullptr) {
                    if (result.clearOutput) tab->output.Text(L"");
                    if (!result.output.empty()) Append(*tab, result.output);
                    cudaFreeBytes_ = freeBytes;
                    RefreshSymbols(work->id);
                    RefreshHeader(work->id);
                }
                if (runningId_ == work->id) runningId_ = -1;
                RefreshStatus();
                const bool close = result.requestExit || pendingClose_.count(work->id) != 0;
                pendingClose_.erase(work->id);
                if (work->done) work->done();
                if (close) {
                    DropQueued(work->id);
                    CloseTabAsync(work->id);
                }
            });
        }
    }).detach();
}

void MainWindow::OnUi(std::function<void()> work) {
    auto gate = std::make_shared<std::promise<void>>();
    auto wait = gate->get_future();
    const bool posted = uiQueue_.TryEnqueue([gate, work = std::move(work)] {
        try {
            work();
        } catch (...) {
        }
        gate->set_value();
    });
    if (!posted) gate->set_value();
    wait.wait();
}

void MainWindow::UpdateActivity() {
    std::set<int> queued;
    {
        std::lock_guard<std::mutex> lock(jobsMutex_);
        for (const auto& job : jobs_) queued.insert(job.id);
    }
    Title(runningId_ >= 0 ? L"CudaMath \u2014 Running" : L"CudaMath");
    for (auto& entry : tabs_) {
        auto& tab = entry.second;
        const bool running = entry.first == runningId_;
        const bool waiting = !running && queued.count(entry.first) != 0;
        if (running) {
            tab.activity.Text(L"Running");
            tab.activity.Foreground(themeBrush(L"AccentFillColorDefaultBrush"));
            tab.activity.Visibility(Visibility::Visible);
        } else if (waiting) {
            tab.activity.Text(L"Waiting");
            tab.activity.Foreground(themeBrush(L"TextFillColorSecondaryBrush"));
            tab.activity.Visibility(Visibility::Visible);
        } else {
            tab.activity.Text(L"");
            tab.activity.Visibility(Visibility::Collapsed);
        }
        if (tab.hint != nullptr && tab.input.Text().empty()) {
            tab.hint.Text(running ? L"Running\u2026" : L"Enter expression here");
            tab.hint.FontStyle(running ? winrt::Windows::UI::Text::FontStyle::Normal
                                       : winrt::Windows::UI::Text::FontStyle::Italic);
        }
    }
}

bool MainWindow::Busy() {
    if (runningId_ >= 0) return true;
    std::lock_guard<std::mutex> lock(jobsMutex_);
    return workerRunning_ || !jobs_.empty();
}

bool MainWindow::DocumentBusy(int id) {
    if (runningId_ == id) return true;
    return Queued(id);
}

bool MainWindow::Queued(int id) {
    std::lock_guard<std::mutex> lock(jobsMutex_);
    for (const auto& job : jobs_) {
        if (job.id == id) return true;
    }
    return false;
}

void MainWindow::DropQueued(int id) {
    std::vector<std::function<void()>> done;
    {
        std::lock_guard<std::mutex> lock(jobsMutex_);
        std::deque<Job> keep;
        for (auto& job : jobs_) {
            if (id < 0 || job.id == id) {
                if (job.done) done.push_back(std::move(job.done));
            } else {
                keep.push_back(std::move(job));
            }
        }
        jobs_ = std::move(keep);
    }
    for (auto& fn : done) fn();
}

void MainWindow::RefreshSymbols(int id) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    std::string selected;
    if (tab->symbols.SelectedItem() != nullptr) {
        selected = symbolName(narrow(unbox_value<hstring>(tab->symbols.SelectedItem())));
    }
    tab->symbols.Items().Clear();
    int selectAt = -1;
    int index = 0;
    for (const auto& entry : tab->session->vars) {
        if (entry.first == selected) selectAt = index;
        tab->symbols.Items().Append(box_value(widen(entry.second.whosLine(entry.first))));
        ++index;
    }
    for (const auto& entry : tab->session->functions) {
        if (entry.first == selected) selectAt = index;
        tab->symbols.Items().Append(box_value(widen(entry.first + "  function")));
        ++index;
    }
    if (selectAt >= 0) tab->symbols.SelectedIndex(selectAt);
    else tab->source.Text(L"");
}

void MainWindow::RefreshHeader(int id) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    tab->title.Text(widen(tab->session->title()));
}

void MainWindow::RefreshStatus() {
    CudaFreeStatus().Text(megabytes(cudaFreeBytes_));
    const auto activeFill = themeBrush(L"AccentFillColorDefaultBrush");
    const auto inactiveFill = themeBrush(L"SubtleFillColorSecondaryBrush");
    const auto activeText = themeBrush(L"TextOnAccentFillColorPrimaryBrush");
    const auto inactiveText = themeBrush(L"TextFillColorSecondaryBrush");
    const auto activeBorder = themeBrush(L"AccentFillColorDefaultBrush");
    const auto inactiveBorder = themeBrush(L"ControlStrongStrokeColorDefaultBrush");
    for (auto& entry : tabs_) {
        auto& tab = entry.second;
        const bool active = entry.first == activeId_;
        tab.titleBar.Background(active ? activeFill : inactiveFill);
        tab.title.Foreground(active ? activeText : inactiveText);
        tab.frame.Background(active ? activeBorder : inactiveBorder);
        tab.frame.Padding(Thickness{1});
        if (entry.first != runningId_) {
            tab.title.Text(widen(tab.session->title()));
            tab.timer.Content(box_value(widen(cudamath::statusTimer(*tab.session))));
            tab.memory.Text(megabytes(tab.session->workspaceBytes()));
        }
    }
    UpdateActivity();
    RebuildWindowMenu();
}

void MainWindow::RebuildWindowMenu() {
    // Clear() notifies a reset, and MenuBarItem ignores that, so the flyout keeps the old entries.
    auto items = WindowMenu().Items();
    while (items.Size() > 0) {
        items.RemoveAt(items.Size() - 1);
    }
    for (const auto& entry : tabs_) {
        MenuFlyoutItem item;
        item.Text(widen(entry.second.session->title()));
        if (entry.first == activeId_) {
            SymbolIcon mark;
            mark.Symbol(Symbol::Accept);
            item.Icon(mark);
        }
        const int id = entry.first;
        item.Click([this, id](IInspectable const&, RoutedEventArgs const&) {
            uiQueue_.TryEnqueue([this, id] {
                Activate(id);
                if (auto* tab = Find(id)) tab->input.Focus(FocusState::Programmatic);
            });
        });
        WindowMenu().Items().Append(item);
    }
}

void MainWindow::RemoveTab(int id) {
    auto* tab = Find(id);
    if (tab == nullptr) return;
    if (DocumentBusy(id)) {
        pendingClose_.insert(id);
        DropQueued(id);
        return;
    }
    cudamath::Session* session = tab->session;
    Border frame = tab->frame;
    tabs_.erase(id);
    uint32_t index = 0;
    if (MdiClient().Children().IndexOf(frame, index)) {
        MdiClient().Children().RemoveAt(index);
    }
    cudamath::Engine::instance().destroySession(*session);
    if (activeId_ == id) {
        activeId_ = -1;
        int next = -1;
        int nextZ = -1;
        for (const auto& entry : tabs_) {
            const int z = Canvas::GetZIndex(entry.second.frame);
            if (z >= nextZ) {
                nextZ = z;
                next = entry.first;
            }
        }
        if (next >= 0) Activate(next);
    }
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

void MainWindow::WindowClosing(winrt::Microsoft::UI::Windowing::AppWindow const&, winrt::Microsoft::UI::Windowing::AppWindowClosingEventArgs const& args) {
    if (allowClose_) return;
    args.Cancel(true);
    if (!closing_) ExitAsync();
}

winrt::fire_and_forget MainWindow::CloseTabAsync(int id) {
    auto lifetime = get_strong();
    auto* tab = Find(id);
    if (tab == nullptr) co_return;
    if (DocumentBusy(id)) {
        pendingClose_.insert(id);
        DropQueued(id);
        co_return;
    }
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
        co_await RunAsync(id, "saveworkspace " + quotePath(path), false);
    } catch (const std::exception& error) {
        MessageBoxW(WindowHandle(), widen(error.what()).c_str(), L"CudaMath", MB_OK | MB_ICONERROR);
        co_return false;
    }
    tab = Find(id);
    if (tab == nullptr) co_return false;
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
    DropQueued(-1);
    co_await WaitUntilIdle();
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
