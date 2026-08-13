add_executable(
  test_pflow test_pflow.cpp
             ${CMAKE_SOURCE_DIR}/tests/unit/utils/test_acopf_utils.cpp
)
target_link_libraries(test_pflow ExaGO::PFLOW)
target_include_directories(
  test_pflow PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/utils
)

if(EXAGO_RUN_TESTS)
  exago_add_test(
    NAME
    "UNIT_TESTS_PFLOW"
    COMMAND
    ${RUNCMD}
    $<TARGET_FILE:test_pflow>
    -validation_dir
    ${CMAKE_SOURCE_DIR}/datafiles/test_validation
    NETFILES
    ${network_files}
  )
endif()
