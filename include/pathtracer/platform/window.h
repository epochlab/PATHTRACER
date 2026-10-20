#pragma once

#include <functional>
#include <string>
#include <utility>

// Forward-declared rather than including <GLFW/glfw3.h>: a consumer needing only the size or the cursor does not pull GLFW in.
struct GLFWwindow;

namespace pathtracer::platform {

// Owns one GLFWwindow and its OpenGL 4.1 core forward-compatible context. glfwInit/glfwTerminate remain the caller's responsibility.
class Window {
public:
    // Sizes the framebuffer, in pixels: the window opens at that many pixels on the display it lands on, and is freely resizable after.
    Window(int framebufferWidth, int framebufferHeight, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&& other) noexcept;
    Window& operator=(Window&& other) noexcept;

    [[nodiscard]] bool shouldClose() const;
    void setShouldClose(bool shouldClose);
    void pollEvents() const;
    void swapBuffers() const;

    // For backends that need the raw GLFW handle (e.g. ImGui's GLFW backend); everything else should use the typed accessors above.
    [[nodiscard]] GLFWwindow* nativeHandle() const noexcept { return window_; }

    // {width, height} of the display viewport in pixels, not screen points: the two differ by 2x on Retina. Never the render resolution.
    [[nodiscard]] std::pair<int, int> framebufferSize() const;

    // {width, height} in screen points, the units cursorPosition() uses. Never divide cursorPosition() by the framebuffer size.
    [[nodiscard]] std::pair<int, int> windowSize() const;

    // Invoked on GLFW key events with GLFW_MOD_* bits; scancode is not forwarded because no consumer needs it. One slot for every hotkey.
    using KeyCallback = std::function<void(int key, int action, int mods)>;
    void setKeyCallback(KeyCallback callback);

    // Polled key state (glfwGetKey), for continuous movement (WASD/QE) where an edge-triggered callback would only fire once per press.
    [[nodiscard]] bool isKeyDown(int key) const;

    // Invoked on GLFW's mouse-button event. Single slot, like KeyCallback.
    using MouseButtonCallback = std::function<void(int button, int action)>;
    void setMouseButtonCallback(MouseButtonCallback callback);

    // Polled cursor position in screen coordinates. Orbit needs only a per-frame delta, so unlike key this needs no callback.
    [[nodiscard]] std::pair<double, double> cursorPosition() const;

    // Hides and locks the cursor to the window (GLFW_CURSOR_DISABLED) while true, as during an LMB-drag orbit.
    void setCursorLocked(bool locked);

private:
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

    GLFWwindow* window_ = nullptr;
    KeyCallback keyCallback_;
    MouseButtonCallback mouseButtonCallback_;
};

}  // namespace pathtracer::platform
