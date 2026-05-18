add_subdirectory(opflow)
add_subdirectory(pflow)
add_subdirectory(ps)

add_executable(test_error_handler test_error_handler.cpp)
target_link_libraries(test_error_handler ExaGO::UTILS)

add_executable(test_logger test_logger.cpp)
target_link_libraries(test_logger ExaGO::UTILS)

if(EXAGO_RUN_TESTS)
  exago_add_test(
    NAME UNIT_TESTS_UTILS COMMAND $<TARGET_FILE:test_error_handler>
  )

  string(REPLACE ";" " " MPICMD_STR "${MPICMD}")
  exago_add_test(
    NAME
    UNIT_TESTS_LOGGER
    COMMAND
    ${CMAKE_CURRENT_SOURCE_DIR}/run_test_logger.pl
    ${MPICMD_STR}
    ${TESTING_MAX_NUMPROCS}
    $<TARGET_FILE:test_logger>
    DEPENDS
    LOGGING
  )
endif()
