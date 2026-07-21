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
include(FetchContent)

# Prefer packages supplied by a toolchain (for example vcpkg), but remain
# buildable on a clean Windows workstation without an external package manager.
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(libzip CONFIG QUIET)
if(NOT libzip_FOUND)
  find_package(ZLIB QUIET)
  if(NOT ZLIB_FOUND)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
    FetchContent_Declare(zlib
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG v1.3.1
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(zlib)
    if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
      add_library(ZLIB::ZLIB ALIAS zlibstatic)
    endif()
    # libzip's CMake script calls FindZLIB even when the zlib target is already
    # present in this build. Supply the in-tree target and headers explicitly.
    set(ZLIB_LIBRARY zlibstatic CACHE FILEPATH "" FORCE)
    set(ZLIB_LIBRARIES zlibstatic)
    set(ZLIB_INCLUDE_DIR "${zlib_SOURCE_DIR}" CACHE PATH "" FORCE)
    set(ZLIB_INCLUDE_DIRS "${zlib_SOURCE_DIR}")
    set(ZLIB_FOUND TRUE)
  endif()

  set(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(BUILD_REGRESS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(ENABLE_BZIP2 OFF CACHE BOOL "" FORCE)
  set(ENABLE_LZMA OFF CACHE BOOL "" FORCE)
  set(ENABLE_ZSTD OFF CACHE BOOL "" FORCE)
  set(ENABLE_OPENSSL OFF CACHE BOOL "" FORCE)
  set(ENABLE_GNUTLS OFF CACHE BOOL "" FORCE)
  set(ENABLE_MBEDTLS OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(libzip
    GIT_REPOSITORY https://github.com/nih-at/libzip.git
    GIT_TAG v1.11.4
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(libzip)
  if(TARGET zip AND NOT TARGET libzip::zip)
    add_library(libzip::zip ALIAS zip)
  endif()
endif()

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
