#include "pathtracer/gfx/shader_program.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

#include <GL/glew.h>

#include "pathtracer/gfx/gl_debug.h"

namespace pathtracer::gfx {

namespace {

std::optional<unsigned int> compileStage(GLenum stage, const std::string& source) {
    unsigned int shader = 0;
    GL_CALL(shader = glCreateShader(stage));
    const char* src = source.c_str();
    GL_CALL(glShaderSource(shader, 1, &src, nullptr));
    GL_CALL(glCompileShader(shader));

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE) {
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        // std::string::data() is valid, non-null and null-terminated even when empty, unlike std::vector<char>::data() at length 0.
        std::string log(static_cast<std::size_t>(logLength), '\0');
        glGetShaderInfoLog(shader, logLength, nullptr, log.data());
        std::cerr << "ShaderProgram: shader compile failed:\n" << log << '\n';
        glDeleteShader(shader);
        return std::nullopt;
    }
    return shader;
}

}  // namespace

std::optional<ShaderProgram> ShaderProgram::loadFromSource(const std::string& vertSrc,
                                                            const std::string& fragSrc) {
    const auto vertShader = compileStage(GL_VERTEX_SHADER, vertSrc);
    if (!vertShader) {
        return std::nullopt;
    }

    const auto fragShader = compileStage(GL_FRAGMENT_SHADER, fragSrc);
    if (!fragShader) {
        glDeleteShader(*vertShader);
        return std::nullopt;
    }

    unsigned int program = 0;
    GL_CALL(program = glCreateProgram());
    GL_CALL(glAttachShader(program, *vertShader));
    GL_CALL(glAttachShader(program, *fragShader));
    GL_CALL(glLinkProgram(program));

    // Shaders are refcounted once attached; safe to delete regardless of link outcome.
    glDeleteShader(*vertShader);
    glDeleteShader(*fragShader);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        GLint logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        std::string log(static_cast<std::size_t>(logLength), '\0');
        glGetProgramInfoLog(program, logLength, nullptr, log.data());
        std::cerr << "ShaderProgram: link failed:\n" << log << '\n';
        glDeleteProgram(program);
        return std::nullopt;
    }

    return ShaderProgram(program);
}

std::optional<std::string> ShaderProgram::readSource(const std::string& path) {
    const std::ifstream file(path);
    if (!file) {
        std::cerr << "ShaderProgram: failed to open " << path << '\n';
        return std::nullopt;
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::optional<ShaderProgram> ShaderProgram::loadFromFiles(const std::string& vertPath,
                                                           const std::string& fragPath) {
    const auto vertSrc = readSource(vertPath);
    if (!vertSrc) {
        return std::nullopt;
    }
    const auto fragSrc = readSource(fragPath);
    if (!fragSrc) {
        return std::nullopt;
    }
    return loadFromSource(*vertSrc, *fragSrc);
}

ShaderProgram::ShaderProgram(unsigned int program) : program_(program) {}

ShaderProgram::~ShaderProgram() {
    if (program_ != 0) {
        glDeleteProgram(program_);
    }
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept
    : program_(std::exchange(other.program_, 0)) {}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    if (this != &other) {
        if (program_ != 0) {
            glDeleteProgram(program_);
        }
        program_ = std::exchange(other.program_, 0);
    }
    return *this;
}

// Not wrapped in GL_CALL: runs every frame (scene shader and display shader are both bound once per frame each).
void ShaderProgram::use() const {
    glUseProgram(program_);
}

int ShaderProgram::uniformLocation(const std::string& name) const {
    const int location = glGetUniformLocation(program_, name.c_str());
    if (location == -1) {
        std::cerr << "ShaderProgram: uniform '" << name << "' not found (program "
                   << program_ << ")\n";
    }
    return location;
}

}  // namespace pathtracer::gfx
