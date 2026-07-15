# compiler/lib/CMakeLists.txt

# ── ARM-IR library ──────────────────────────────────────────────────────────
add_library(armcc_ir STATIC
  arm_ir/types.cpp
  arm_ir/ops.cpp
  arm_ir/graph.cpp
)
target_include_directories(armcc_ir PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/compiler/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(armcc_ir PUBLIC nlohmann_json::nlohmann_json)

# ── Analysis library ─────────────────────────────────────────────────────────
add_library(armcc_analysis STATIC
  analysis/cost_model.cpp
  analysis/graph_analyzer.cpp
)
target_include_directories(armcc_analysis PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/compiler/include>
)
target_link_libraries(armcc_analysis PUBLIC armcc_ir nlohmann_json::nlohmann_json)

# ── Passes library ───────────────────────────────────────────────────────────
add_library(armcc_passes STATIC
  passes/pass_manager.cpp
  passes/graph/graph_passes.cpp
  passes/graph/hardware_fusion.cpp
  passes/quantization/quant_passes.cpp
  passes/memory/memory_passes.cpp
)
target_include_directories(armcc_passes PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/compiler/include>
)
target_link_libraries(armcc_passes PUBLIC armcc_ir armcc_analysis)

# ── Generator library ────────────────────────────────────────────────────────
add_library(armcc_generator STATIC
  generator/graph_family_generator.cpp
  generator/graph_selector.cpp
  generator/condition_matrix.cpp
)
target_include_directories(armcc_generator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/compiler/include>
)
target_link_libraries(armcc_generator PUBLIC armcc_ir armcc_analysis armcc_passes)

# ── Memory library ───────────────────────────────────────────────────────────
add_library(armcc_memory STATIC
  memory/memory_planner.cpp
  memory/tensor_allocator.cpp
  memory/kv_cache_layout.cpp
  memory/mmap_layout.cpp
)
target_include_directories(armcc_memory PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/compiler/include>
)
target_link_libraries(armcc_memory PUBLIC armcc_ir armcc_analysis)

# ── Package library ──────────────────────────────────────────────────────────
add_library(armcc_package STATIC
  package/package_generator.cpp
  package/manifest_writer.cpp
)
target_include_directories(armcc_package PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/compiler/include>
)
target_link_libraries(armcc_package PUBLIC
  armcc_ir
  armcc_generator
  armcc_memory
  nlohmann_json::nlohmann_json
  libzip::zip
)

# ── All-in-one convenience target ────────────────────────────────────────────
add_library(armcc_all INTERFACE)
target_link_libraries(armcc_all INTERFACE
  armcc_ir
  armcc_analysis
  armcc_passes
  armcc_generator
  armcc_memory
  armcc_package
)
