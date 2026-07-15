# tests/CMakeLists.txt

find_package(GTest REQUIRED)
include(GoogleTest)

# ── Unit tests ───────────────────────────────────────────────────────────────
add_executable(armcc_unit_tests
  unit/test_ir_types.cpp
  unit/test_ir_graph.cpp
  unit/test_pass_manager.cpp
  unit/test_cost_model.cpp
)
target_link_libraries(armcc_unit_tests PRIVATE
  armcc_all
  GTest::gtest_main
)
target_include_directories(armcc_unit_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/compiler/include
  ${CMAKE_BINARY_DIR}/include
)
gtest_discover_tests(armcc_unit_tests)
