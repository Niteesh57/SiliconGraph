cmake_minimum_required(VERSION 3.25)
project(ARMAICompiler
  VERSION 0.1.0
  DESCRIPTION "Universal AI Model Optimizer for ARM Devices"
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ────────────────────────────────────────────────────────────────────
option(ARMCC_ENABLE_MLIR    "Enable MLIR/LLVM compiler backend"  ON)
option(ARMCC_ENABLE_TESTS   "Build unit and integration tests"   ON)
option(ARMCC_ENABLE_VULKAN  "Enable Vulkan compute kernel gen"   ON)
option(ARMCC_ENABLE_OPENCL  "Enable OpenCL kernel generation"    ON)
option(ARMCC_ENABLE_METAL   "Enable Metal (Apple) kernel gen"    OFF)
option(ARMCC_ENABLE_HTP     "Enable Qualcomm HTP (Hexagon) delegate" OFF)
option(ARMCC_ENABLE_NEURIPILOT "Enable MediaTek NeuroPilot delegate" OFF)
option(ARMCC_ENABLE_KLEIDIAI "Build with Arm KleidiAI CPU micro-kernels" OFF)
option(ARMCC_FETCH_KLEIDIAI "Fetch the pinned KleidiAI release when needed" ON)
set(ARMCC_KLEIDIAI_SOURCE_DIR "" CACHE PATH
    "Optional path to a checked-out KleidiAI source tree")

# ── Dependencies ───────────────────────────────────────────────────────────────
find_package(nlohmann_json REQUIRED)
find_package(flatbuffers   REQUIRED)
find_package(libzip        REQUIRED)

if(ARMCC_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIR in: ${MLIR_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
endif()

if(ARMCC_ENABLE_VULKAN)
  find_package(Vulkan)
  if(Vulkan_FOUND)
    message(STATUS "Vulkan found — enabling GPU kernel generation")
  else()
    message(WARNING "Vulkan not found — GPU kernel generation disabled")
    set(ARMCC_ENABLE_VULKAN OFF)
  endif()
endif()

# ── Arm KleidiAI CPU micro-kernels ─────────────────────────────────────────
# KleidiAI is Arm-only. Keep it optional so host-side x86 development and
# compiler-only builds do not attempt to compile Arm assembly sources.
add_library(armcc_kleidiai INTERFACE)
if(ARMCC_ENABLE_KLEIDIAI)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARMCC_SYSTEM_PROCESSOR_LOWER)
  if(NOT ARMCC_SYSTEM_PROCESSOR_LOWER MATCHES "^(aarch64|arm64|armv[0-9]+)")
    message(FATAL_ERROR
      "ARMCC_ENABLE_KLEIDIAI requires an Arm target. Configure with an Arm "
      "toolchain, for example Android arm64-v8a.")
  endif()

  if(ARMCC_KLEIDIAI_SOURCE_DIR)
    if(NOT EXISTS "${ARMCC_KLEIDIAI_SOURCE_DIR}/CMakeLists.txt")
      message(FATAL_ERROR
        "ARMCC_KLEIDIAI_SOURCE_DIR does not contain a KleidiAI CMake project: "
        "${ARMCC_KLEIDIAI_SOURCE_DIR}")
    endif()
    set(ARMCC_KLEIDIAI_DIR "${ARMCC_KLEIDIAI_SOURCE_DIR}")
  elseif(ARMCC_FETCH_KLEIDIAI)
    include(FetchContent)
    FetchContent_Declare(kleidiai
      GIT_REPOSITORY https://github.com/ARM-software/kleidiai.git
      GIT_TAG v1.28.0
      GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(kleidiai)
    if(NOT kleidiai_POPULATED)
      set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
      set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
      FetchContent_Populate(kleidiai)
    endif()
    set(ARMCC_KLEIDIAI_DIR "${kleidiai_SOURCE_DIR}")
  else()
    message(FATAL_ERROR
      "Set ARMCC_KLEIDIAI_SOURCE_DIR or enable ARMCC_FETCH_KLEIDIAI to use "
      "KleidiAI.")
  endif()

  # Do not build KleidiAI's tests/benchmarks as part of SiliconGraph.
  set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  add_subdirectory("${ARMCC_KLEIDIAI_DIR}"
                   "${CMAKE_BINARY_DIR}/_deps/kleidiai-build"
                   EXCLUDE_FROM_ALL)
  target_link_libraries(armcc_kleidiai INTERFACE KleidiAI::kleidiai)
  target_compile_definitions(armcc_kleidiai INTERFACE ARMCC_HAS_KLEIDIAI=1)
  message(STATUS "KleidiAI enabled for Arm CPU kernel integration")
else()
  target_compile_definitions(armcc_kleidiai INTERFACE ARMCC_HAS_KLEIDIAI=0)
endif()

# ── Compiler flags ─────────────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O0 -g>
  )
endif()

# ── Version header ─────────────────────────────────────────────────────────────
configure_file(
  "${CMAKE_SOURCE_DIR}/compiler/include/armcc/version.h.in"
  "${CMAKE_BINARY_DIR}/include/armcc/version.h"
  @ONLY
)

# ── Subdirectories ─────────────────────────────────────────────────────────────
add_subdirectory(compiler/lib)
add_subdirectory(compiler/tools)

if(ARMCC_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── Install ────────────────────────────────────────────────────────────────────
install(DIRECTORY compiler/include/ DESTINATION include)
install(TARGETS armcc arm-ir-dump DESTINATION bin)
