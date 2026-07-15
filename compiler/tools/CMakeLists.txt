# compiler/tools/CMakeLists.txt

# ── armcc CLI ────────────────────────────────────────────────────────────────
add_executable(armcc armcc/main.cpp)
target_link_libraries(armcc PRIVATE armcc_all nlohmann_json::nlohmann_json)
target_include_directories(armcc PRIVATE
  ${CMAKE_SOURCE_DIR}/compiler/include
  ${CMAKE_BINARY_DIR}/include
)
install(TARGETS armcc DESTINATION bin)

# ── arm-ir-dump tool ─────────────────────────────────────────────────────────
add_executable(arm-ir-dump arm-ir-dump/main.cpp)
target_link_libraries(arm-ir-dump PRIVATE armcc_ir nlohmann_json::nlohmann_json)
target_include_directories(arm-ir-dump PRIVATE
  ${CMAKE_SOURCE_DIR}/compiler/include
  ${CMAKE_BINARY_DIR}/include
)
install(TARGETS arm-ir-dump DESTINATION bin)
