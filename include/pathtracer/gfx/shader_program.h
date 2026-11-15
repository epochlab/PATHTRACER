#pragma once

#include <optional>
#include <string>

namespace pathtracer::gfx {

// Owns one linked GL program object, move-only. Compile and link failure returns nullopt rather than exiting: the caller decides.
class ShaderProgram {
public:
    ~ShaderProgram();

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;
    ShaderProgram(ShaderProgram&& other) noexcept;
    ShaderProgram& operator=(ShaderProgram&& other) noexcept;

    // A GLSL file whole, or nullopt with the path logged; shared by every caller that splices a file into generated source.
    static std::optional<std::string> readSource(const std::string& path);

    // The one compile/link/error-check: OCIO's runtime-generated GLSL never exists as a file, so sources are strings.
    static std::optional<ShaderProgram> loadFromSource(const std::string& vertSrc,
                                                        const std::string& fragSrc);

    void use() const;

    // -1, GL's own sentinel, if name matches no active uniform. Not cached: each is looked up once at startup.
    [[nodiscard]] int uniformLocation(const std::string& name) const;

private:
    explicit ShaderProgram(unsigned int program);

    unsigned int program_ = 0;
};

}  // namespace pathtracer::gfx
