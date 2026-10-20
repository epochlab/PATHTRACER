#include "pathtracer/platform/window.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

#include <GLFW/glfw3.h>

namespace pathtracer::platform {

Window::Window(int framebufferWidth, int framebufferHeight, const std::string& title) {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);  // mandatory on macOS
    // Deliberately not GLFW_SRGB_CAPABLE: display encoding happens only in the OCIO shader, never in a driver-level conversion.

    // Stated, not left to the per-platform default: the framebuffer is a HiDPI backing store, which the point size below divides out.
    glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // shown after the resize below, so the opening size is never seen twice

    window_ = glfwCreateWindow(framebufferWidth, framebufferHeight, title.c_str(), nullptr, nullptr);
    if (window_ == nullptr) {
        std::cerr << "Window: glfwCreateWindow failed\n";
        std::exit(EXIT_FAILURE);
    }

    // Scale of the display the window actually opened on, not a guessed monitor's; fractional scaling can leave it a pixel off.
    float xscale = 1.0F;
    float yscale = 1.0F;
    glfwGetWindowContentScale(window_, &xscale, &yscale);
    glfwSetWindowSize(window_, static_cast<int>(std::lround(static_cast<double>(framebufferWidth) / xscale)),
                      static_cast<int>(std::lround(static_cast<double>(framebufferHeight) / yscale)));
    glfwShowWindow(window_);

    glfwMakeContextCurrent(window_);
    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, &Window::keyCallback);
    glfwSetMouseButtonCallback(window_, &Window::mouseButtonCallback);
}

Window::~Window() {
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
    }
}

Window::Window(Window&& other) noexcept
    : window_(std::exchange(other.window_, nullptr)),
      keyCallback_(std::move(other.keyCallback_)),
      mouseButtonCallback_(std::move(other.mouseButtonCallback_)) {
    if (window_ != nullptr) {
        glfwSetWindowUserPointer(window_, this);
    }
}

Window& Window::operator=(Window&& other) noexcept {
    if (this != &other) {
        if (window_ != nullptr) {
            glfwDestroyWindow(window_);
        }
        window_ = std::exchange(other.window_, nullptr);
        keyCallback_ = std::move(other.keyCallback_);
        mouseButtonCallback_ = std::move(other.mouseButtonCallback_);
        if (window_ != nullptr) {
            glfwSetWindowUserPointer(window_, this);
        }
    }
    return *this;
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(window_) != 0;
}

void Window::setShouldClose(bool shouldClose) {
    glfwSetWindowShouldClose(window_, shouldClose ? GLFW_TRUE : GLFW_FALSE);
}

void Window::pollEvents() const {
    glfwPollEvents();
}

void Window::swapBuffers() const {
    glfwSwapBuffers(window_);
}

std::pair<int, int> Window::framebufferSize() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    return {width, height};
}

std::pair<int, int> Window::windowSize() const {
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window_, &width, &height);
    return {width, height};
}

void Window::setKeyCallback(KeyCallback callback) {
    keyCallback_ = std::move(callback);
}

void Window::keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int mods) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self != nullptr && self->keyCallback_) {
        self->keyCallback_(key, action, mods);
    }
}

bool Window::isKeyDown(int key) const {
    return glfwGetKey(window_, key) == GLFW_PRESS;
}

void Window::setMouseButtonCallback(MouseButtonCallback callback) {
    mouseButtonCallback_ = std::move(callback);
}

std::pair<double, double> Window::cursorPosition() const {
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(window_, &x, &y);
    return {x, y};
}

void Window::setCursorLocked(bool locked) {
    glfwSetInputMode(window_, GLFW_CURSOR, locked ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

void Window::mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self != nullptr && self->mouseButtonCallback_) {
        self->mouseButtonCallback_(button, action);
    }
}

}  // namespace pathtracer::platform
