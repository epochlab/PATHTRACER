#include "pathtracer/scene/gltf_loader.h"

#include <cgltf.h>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace pathtracer::scene {

namespace {

// Raw glTF-read vertex, one per accessor entry: an intermediate the triangle builders consume, not retained past load.
struct Vertex {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec3 normal;
    glm::vec4 tangent;  // .w = bitangent handedness (glTF convention)
    glm::vec3 colour;   // COLOR_0, multiplies baseColor; white (1,1,1) when the primitive has none
};

glm::mat4 localNodeTransform(const cgltf_node* node) {
    if (node->has_matrix) {
        return glm::make_mat4(node->matrix);
    }
    float local[16];
    cgltf_node_transform_local(node, local);
    return glm::make_mat4(local);
}

// Appends this primitive's triangles baked to world space: EmbreeAccel operates on one flat soup, not per-instance geometry.
void appendWorldTriangles(const std::vector<Vertex>& vertices,
                           const std::vector<unsigned int>& indices, const glm::mat4& transform,
                           std::vector<Triangle>& outWorldTriangles) {
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        const auto toWorld = [&](unsigned int index) {
            return glm::vec3(transform * glm::vec4(vertices[index].position, 1.0F));
        };
        outWorldTriangles.push_back(
            Triangle{toWorld(indices[i]), toWorld(indices[i + 1]), toWorld(indices[i + 2])});
    }
}

// Parallel to appendWorldTriangles: normal via inverse-transpose, tangent via transform directly, handedness = sign(det M).
void appendShadingTriangles(const std::vector<Vertex>& vertices,
                             const std::vector<unsigned int>& indices, const glm::mat4& transform,
                             float handedness, int instanceIndex, std::vector<ShadingTriangle>& outShadingTriangles) {
    const glm::mat3 linear(transform);
    const glm::mat3 normalMatrix = glm::inverseTranspose(linear);
    const auto toWorldVertex = [&](unsigned int index) {
        const Vertex& v = vertices[index];
        return ShadingVertex{
            glm::vec3(transform * glm::vec4(v.position, 1.0F)),
            glm::normalize(normalMatrix * v.normal),
            v.uv,
            // sign det[M^-T N, MT, MB] = sign(det M): cross(N',T') tracks M*B only once w takes the determinant's sign.
            glm::vec4(glm::normalize(linear * glm::vec3(v.tangent)), v.tangent.w * handedness),
            v.colour,
        };
    };
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        outShadingTriangles.push_back(ShadingTriangle{
            toWorldVertex(indices[i]),
            toWorldVertex(indices[i + 1]),
            toWorldVertex(indices[i + 2]),
            instanceIndex,
        });
    }
}

struct RequiredAccessors {
    const cgltf_accessor* position;
    const cgltf_accessor* normal;
    const cgltf_accessor* uv;
    const cgltf_accessor* tangent;
    const cgltf_accessor* color;  // COLOR_0, optional -- nullptr means "no vertex colour"
};

// Locates the required accessors plus an optional COLOR_0; nullopt for a missing or sparse one, which cgltf cannot report failing.
std::optional<RequiredAccessors> findAttributeAccessors(const cgltf_primitive& prim) {
    RequiredAccessors acc{nullptr, nullptr, nullptr, nullptr, nullptr};
    for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
        const cgltf_attribute& attr = prim.attributes[ai];
        if (attr.type == cgltf_attribute_type_position) {
            acc.position = attr.data;
        } else if (attr.type == cgltf_attribute_type_normal) {
            acc.normal = attr.data;
        } else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) {
            acc.uv = attr.data;
        } else if (attr.type == cgltf_attribute_type_tangent) {
            acc.tangent = attr.data;
        } else if (attr.type == cgltf_attribute_type_color && attr.index == 0) {
            acc.color = attr.data;
        }
    }
    if (acc.position == nullptr || acc.normal == nullptr || acc.uv == nullptr ||
        acc.tangent == nullptr) {
        std::cerr << "loadGltf: primitive missing position/normal/uv/tangent\n";
        return std::nullopt;
    }
    if (acc.position->is_sparse || acc.normal->is_sparse || acc.uv->is_sparse ||
        acc.tangent->is_sparse || (acc.color != nullptr && acc.color->is_sparse)) {
        std::cerr << "loadGltf: sparse accessors are not supported\n";
        return std::nullopt;
    }
    return acc;
}

std::vector<Vertex> readVertices(const RequiredAccessors& acc) {
    std::vector<Vertex> vertices(acc.position->count);
    for (cgltf_size vi = 0; vi < acc.position->count; ++vi) {
        Vertex& v = vertices[vi];
        cgltf_accessor_read_float(acc.position, vi, &v.position.x, 3);
        cgltf_accessor_read_float(acc.normal, vi, &v.normal.x, 3);
        // No V flip: glTF's v=0-at-top already matches loadExr's row-0-at-top convention.
        cgltf_accessor_read_float(acc.uv, vi, &v.uv.x, 2);
        cgltf_accessor_read_float(acc.tangent, vi, &v.tangent.x, 4);
        if (acc.color != nullptr) {
        // COLOR_0 may be VEC3 or VEC4 and cgltf defaults no 4th component, so the read is sized to the accessor; alpha is dropped.
            float raw[4] = {1.0F, 1.0F, 1.0F, 1.0F};
            cgltf_accessor_read_float(acc.color, vi, raw, cgltf_num_components(acc.color->type));
            v.colour = glm::vec3(raw[0], raw[1], raw[2]);
        } else {
            v.colour = glm::vec3(1.0F);
        }
    }
    return vertices;
}

// Rejects a sparse index accessor for the same reason; a missing one means the implied 0..vertexCount-1 (glTF 2.0 3.7.2.1).
std::optional<std::vector<unsigned int>> readIndices(const cgltf_accessor* indicesAcc, cgltf_size vertexCount) {
    if (indicesAcc == nullptr) {
        std::vector<unsigned int> indices(vertexCount);
        std::iota(indices.begin(), indices.end(), 0U);
        return indices;
    }
    if (indicesAcc->is_sparse) {
        std::cerr << "loadGltf: sparse accessors are not supported\n";
        return std::nullopt;
    }
    std::vector<unsigned int> indices(indicesAcc->count);
    for (cgltf_size ii = 0; ii < indicesAcc->count; ++ii) {
        indices[ii] = static_cast<unsigned int>(cgltf_accessor_read_index(indicesAcc, ii));
    }
    return indices;
}

// Expands a strip or fan to a list in glTF 2.0 3.7.2.1's per-triangle vertex order, which is what fixes each face's winding.
std::vector<unsigned int> toTriangleList(std::vector<unsigned int> indices, cgltf_primitive_type type) {
    if (type == cgltf_primitive_type_triangles) {
        return indices;
    }
    const std::size_t triangleCount = indices.size() < 3 ? 0 : indices.size() - 2;
    std::vector<unsigned int> list;
    list.reserve(3 * triangleCount);
    if (type == cgltf_primitive_type_triangle_strip) {
        for (std::size_t i = 0; i < triangleCount; ++i) {
            list.insert(list.end(), {indices[i], indices[i + 1 + (i % 2)], indices[i + 2 - (i % 2)]});
        }
    } else {
        for (std::size_t i = 0; i < triangleCount; ++i) {
            list.insert(list.end(), {indices[i + 1], indices[i + 2], indices[0]});
        }
    }
    return list;
}

// True for the triangle topologies; points and lines have no area, so no ray can hit them and they carry no surface.
bool isSurface(cgltf_primitive_type type) {
    return type == cgltf_primitive_type_triangles || type == cgltf_primitive_type_triangle_strip ||
           type == cgltf_primitive_type_triangle_fan;
}

// Builds one MeshInstance from a triangle-topology primitive, failing with nullopt rather than defaulting geometry it cannot read.
std::optional<MeshInstance> loadPrimitive(const cgltf_primitive& prim, const glm::mat4& transform, int instanceIndex,
                                           const std::string& name,
                                           std::vector<Triangle>& outWorldTriangles,
                                           std::vector<ShadingTriangle>& outShadingTriangles) {
    const std::optional<RequiredAccessors> acc = findAttributeAccessors(prim);
    if (!acc.has_value()) {
        return std::nullopt;
    }
    std::optional<std::vector<unsigned int>> stream = readIndices(prim.indices, acc->position->count);
    if (!stream.has_value()) {
        return std::nullopt;
    }
    std::vector<unsigned int> indices = toTriangleList(std::move(*stream), prim.type);
    // glTF 2.0 3.7.2.1: a negative global determinant makes faces clockwise; re-wind so cross(e1,e2) keeps facing the normal.
    const float handedness = glm::determinant(glm::mat3(transform)) < 0.0F ? -1.0F : 1.0F;
    if (handedness < 0.0F) {
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            std::swap(indices[i + 1], indices[i + 2]);
        }
    }
    const std::vector<Vertex> vertices = readVertices(*acc);
    appendWorldTriangles(vertices, indices, transform, outWorldTriangles);
    appendShadingTriangles(vertices, indices, transform, handedness, instanceIndex, outShadingTriangles);

    // Geometry only: every slot starts at its neutral default, and bindSceneTextures binds the scene JSON's maps by node name.
    return MeshInstance{
        Material{},
        transform,
        name,
    };
}

// Hard cap on node-graph recursion. glTF node hierarchies are untrusted and cgltf_validate checks neither cycles nor depth.
constexpr int kMaxNodeDepth = 256;

// A node hierarchy is a tree, so recursion is its structure; the depth cap above is what makes it safe on untrusted input.

// NOLINTNEXTLINE(misc-no-recursion)
bool walkNodes(cgltf_node* const* nodes, cgltf_size count, const glm::mat4& parentTransform,
               std::vector<MeshInstance>& instances, std::vector<Triangle>& worldTriangles,
               std::vector<ShadingTriangle>& shadingTriangles, int depth = 0) {
    if (depth >= kMaxNodeDepth) {
        std::cerr << "loadGltf: node hierarchy exceeds max depth " << kMaxNodeDepth
                   << " (cyclic or pathologically nested)\n";
        return false;
    }
    for (cgltf_size ni = 0; ni < count; ++ni) {
        const cgltf_node* node = nodes[ni];
        const glm::mat4 world = parentTransform * localNodeTransform(node);

        if (node->mesh != nullptr) {
            const std::string name = node->name != nullptr ? node->name : "";
            for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
                const cgltf_primitive& prim = node->mesh->primitives[pi];
                if (!isSurface(prim.type)) {
                    std::cerr << "loadGltf: skipping a point or line primitive on node '" << name << "'\n";
                    continue;
                }
                const int instanceIndex = static_cast<int>(instances.size());  // index this primitive's MeshInstance will get
                std::optional<MeshInstance> instance =
                    loadPrimitive(prim, world, instanceIndex, name, worldTriangles, shadingTriangles);
                if (!instance.has_value()) {
                    return false;
                }
                instances.push_back(std::move(*instance));
            }
        }

        if (!walkNodes(node->children, node->children_count, world, instances, worldTriangles, shadingTriangles,
                       depth + 1)) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::optional<LoadedModel> loadGltf(const std::string& path, const glm::mat4& rootTransform) {
    const cgltf_options options{};
    cgltf_data* data = nullptr;

    if (cgltf_parse_file(&options, path.c_str(), &data) != cgltf_result_success) {
        std::cerr << "loadGltf: failed to parse " << path << '\n';
        return std::nullopt;
    }
    if (cgltf_load_buffers(&options, data, path.c_str()) != cgltf_result_success) {
        std::cerr << "loadGltf: failed to load buffers for " << path << '\n';
        cgltf_free(data);
        return std::nullopt;
    }
    if (cgltf_validate(data) != cgltf_result_success) {
        std::cerr << "loadGltf: validation failed for " << path << '\n';
        cgltf_free(data);
        return std::nullopt;
    }

    LoadedModel model;
    const bool ok = data->scene != nullptr &&
                    walkNodes(data->scene->nodes, data->scene->nodes_count, rootTransform, model.instances,
                              model.worldTriangles, model.shadingTriangles);

    cgltf_free(data);

    if (!ok || model.instances.empty()) {
        std::cerr << "loadGltf: no renderable primitives found in " << path << '\n';
        return std::nullopt;
    }
    return model;
}

void appendQuadLights(LoadedModel& model, const std::vector<QuadLight>& lights,
                       std::vector<int>& instanceLightIndex) {
    for (std::size_t i = 0; i < lights.size(); ++i) {
        const QuadLight& light = lights[i];
        const glm::vec3& normal = light.normal;
        const glm::vec4 tangent(glm::normalize(light.edge0), 1.0F);
        const auto vertex = [&](const glm::vec3& position, glm::vec2 uv) {
            return ShadingVertex{position, normal, uv, tangent};
        };
        // Corners p00/p10/p01/p11 split along the p00-p11 diagonal, both wound so cross(v1-v0, v2-v0) reproduces `normal`.
        const ShadingVertex p00 = vertex(light.origin, glm::vec2(0.0F, 0.0F));
        const ShadingVertex p10 = vertex(light.origin + light.edge0, glm::vec2(1.0F, 0.0F));
        const ShadingVertex p01 = vertex(light.origin + light.edge1, glm::vec2(0.0F, 1.0F));
        const ShadingVertex p11 = vertex(light.origin + light.edge0 + light.edge1, glm::vec2(1.0F, 1.0F));

        const int instanceIndex = static_cast<int>(model.instances.size());
        model.worldTriangles.push_back(Triangle{p00.position, p10.position, p11.position});
        model.worldTriangles.push_back(Triangle{p00.position, p11.position, p01.position});
        model.shadingTriangles.push_back(ShadingTriangle{p00, p10, p11, instanceIndex});
        model.shadingTriangles.push_back(ShadingTriangle{p00, p11, p01, instanceIndex});
        model.instances.push_back(MeshInstance{Material{}, glm::mat4(1.0F),
                                                 "__quadLight" + std::to_string(i)});
        instanceLightIndex.push_back(static_cast<int>(i));
    }
}

}  // namespace pathtracer::scene
