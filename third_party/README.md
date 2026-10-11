# Third-party dependencies

Vendored here rather than resolved by a package manager, so a checkout builds without a fetch step.

**The rule:** a single-header library is copied in as that one file; a multi-file library is a git submodule.
Copying a header is the whole dependency and a submodule for one file is pure plumbing, but copying a source
tree means hand-tracking upstream, which a submodule pin does for free. That is why imgui and MikkTSpace (a `.c` plus its
`.h`) are the two entries in `.gitmodules`.

| Library | Version | Upstream | Vendored as | Licence |
|---|---|---|---|---|
| cgltf | 1.15 | https://github.com/jkuhlmann/cgltf | `cgltf/cgltf.h`, single header | MIT |
| nlohmann/json | 3.11.3 | https://github.com/nlohmann/json | `nlohmann/json.hpp`, single header | MIT |
| Dear ImGui | v1.92.9 (`01380c5`) | https://github.com/ocornut/imgui | submodule, `shallow = true` | MIT |
| MikkTSpace | 1.0 (`3e895b4`) | https://github.com/mmikk/MikkTSpace | submodule, `shallow = true` | zlib |

Versions are read from the sources themselves: `cgltf.h`'s banner comment, `json.hpp`'s
`NLOHMANN_JSON_VERSION_*` macros, the imgui submodule's tag, and `mikktspace.h`'s `Version:` line plus the pinned commit. Update a single-header library by replacing
the file and editing the row above; update a submodule with `git submodule update --remote third_party/<name>`.

Only six of imgui's sources are compiled (`imgui{,_draw,_tables,_widgets}.cpp` plus the GLFW and OpenGL3
backends); the rest of the submodule tree is unused.

Build-system dependencies resolved externally, not vendored: Embree 4, OpenEXR, Imath, OpenColorIO, OpenImageIO 3,
GLFW 3.5, GLEW, glm, OpenGL, zlib.

Licence texts for the four vendored libraries are in `NOTICE`.
