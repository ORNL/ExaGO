# Tests with "analytical" solutions
add_subdirectory(objective)
add_subdirectory(constraint_jacobian)

# Tests using PETSc solution as reference
if(EXAGO_ENABLE_RAJA AND EXAGO_ENABLE_CUDA)
  set_source_files_properties(test_acopf.cpp PROPERTIES LANGUAGE CUDA)
  if(EXAGO_ENABLE_HIOP_SPARSE)
    set_source_files_properties(test_eqjac_gpu.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(test_ineqjac_gpu.cpp PROPERTIES LANGUAGE CUDA)
  endif()
endif()

add_executable(test_acopf test_acopf.cpp
               ${CMAKE_SOURCE_DIR}/tests/unit/utils/test_acopf_utils.cpp)
target_link_libraries(test_acopf ExaGO::OPFLOW)
target_include_directories(test_acopf PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/utils)

if(EXAGO_ENABLE_HIOP_SPARSE AND EXAGO_ENABLE_GPU)
  add_executable(test_ineqjac_gpu test_ineqjac_gpu.cpp)
  target_link_libraries(test_ineqjac_gpu ExaGO::OPFLOW)
  target_include_directories(test_ineqjac_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src/opflow)

  add_executable(test_eqjac_gpu test_eqjac_gpu.cpp)
  target_link_libraries(test_eqjac_gpu ExaGO::OPFLOW)
  target_include_directories(test_eqjac_gpu PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/opflow)
endif()

if(EXAGO_RUN_TESTS)
  exago_add_test(
    NAME
    "UNIT_TESTS_OPFLOW"
    DEPENDS
    HIOP
    COMMAND
    ${RUNCMD}
    $<TARGET_FILE:test_acopf>
    -opflow_genbusvoltage
    VARIABLE_WITHIN_BOUNDS
    -validation_dir
    ${CMAKE_SOURCE_DIR}/datafiles/test_validation
    NETFILES
    ${network_files}
  )

  exago_add_test(
    NAME
    "UNIT_TESTS_OPFLOW_EQUALITY_CONSTRAINT_JACOBIAN_GPU"
    DEPENDS
    GPU
    HIOP_SPARSE
    RAJA
    COMMAND
    ${RUNCMD}
    $<TARGET_FILE:test_eqjac_gpu>
    NETFILES
    ${network_files}
  )

  exago_add_test(
    NAME
    "UNIT_TESTS_OPFLOW_INEQUALITY_CONSTRAINT_JACOBIAN_GPU"
    DEPENDS
    GPU
    HIOP_SPARSE
    RAJA
    COMMAND
    ${RUNCMD}
    $<TARGET_FILE:test_ineqjac_gpu>
    -opflow_genbusvoltage
    VARIABLE_WITHIN_BOUNDS
    NETFILES
    ${network_files}
  )
endif()
