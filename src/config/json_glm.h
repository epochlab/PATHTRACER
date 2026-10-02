#pragma once

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

// ADL hook (found via glm's namespace) so nlohmann can do j.at("key").get<glm::vec3>() instead of indexing components by hand.
namespace glm {

inline void from_json(const nlohmann::json& j, vec2& v) {
    if (!j.is_array() || j.size() != 2) {
        throw nlohmann::json::type_error::create(302, "expected a 2-element array for glm::vec2", &j);
    }
    v = vec2{j[0].get<float>(), j[1].get<float>()};
}

inline void from_json(const nlohmann::json& j, vec3& v) {
    if (!j.is_array() || j.size() != 3) {
        throw nlohmann::json::type_error::create(302, "expected a 3-element array for glm::vec3", &j);
    }
    v = vec3{j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
}

}  // namespace glm
