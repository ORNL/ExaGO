find_package(Python)
if(NOT Python_FOUND)
  return()
endif()

if(EXAGO_ENABLE_RAJA AND EXAGO_ENABLE_CUDA)
  set_source_files_properties(jac_eq_acopf.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(test_eqjac_compare.cpp PROPERTIES LANGUAGE CUDA)
endif()
if(EXAGO_ENABLE_RAJA AND EXAGO_ENABLE_HIP)
  set_source_files_properties(test_eqjac_compare.cpp PROPERTIES LANGUAGE HIP)
endif()

add_executable(test_eqjac_compare test_eqjac_compare.cpp)
target_link_libraries(test_eqjac_compare ExaGO::OPFLOW)
target_include_directories(
  test_eqjac_compare PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/opflow
)

if(EXAGO_ENABLE_RAJA AND EXAGO_ENABLE_HIP)
  set_source_files_properties(test_eqjac_perf.cpp PROPERTIES LANGUAGE HIP)
endif()
if(EXAGO_ENABLE_RAJA AND EXAGO_ENABLE_CUDA)
  set_source_files_properties(test_eqjac_perf.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable(test_eqjac_perf test_eqjac_perf.cpp)
target_link_libraries(test_eqjac_perf ExaGO::OPFLOW)
target_include_directories(
  test_eqjac_perf PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/opflow
)

add_executable(
  jac_eq_acopf jac_eq_acopf.cpp
               ${CMAKE_SOURCE_DIR}/tests/unit/utils/test_acopf_utils.cpp
)
target_link_libraries(jac_eq_acopf ExaGO::OPFLOW)
target_include_directories(
  jac_eq_acopf PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/utils
)
target_include_directories(
  jac_eq_acopf PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/opflow
)

# Network files to run on - doing 3 and 600 bus examples.
set(prefix ${CMAKE_SOURCE_DIR}/datafiles/unit/opflow/objective/)
# Map num_copies to a specific netfile
set(num_copies 1 3 600)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cecj.csv ${CMAKE_CURRENT_BINARY_DIR}/cecj.csv
  COPYONLY
)
foreach(n IN LISTS num_copies)
  execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_network.py ${n}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endforeach()

set(solvers "")
set(models "")
set(dependencies "")

# Set up Ipopt solver tests if enabled
if(EXAGO_ENABLE_IPOPT)
  list(APPEND solvers IPOPT)
  list(APPEND models POWER_BALANCE_POLAR)
  list(APPEND dependencies IPOPT)
endif()

# Set up HIOPSPARSE solver tests if enabled
if(TARGET HiOp::SPARSE)
  list(APPEND solvers HIOPSPARSE)
  list(APPEND models POWER_BALANCE_POLAR)
  list(APPEND dependencies HIOP_SPARSE)
endif()

if(EXAGO_RUN_TESTS)
  if(EXAGO_ENABLE_RAJA AND EXAGO_ENABLE_HIOP)
    set(prefix_net ${CMAKE_SOURCE_DIR}/datafiles/unit/opflow/objective/)
    exago_add_test(
      NAME
      UNIT_TESTS_EQJAC_GPU_VS_PETSC_testx3
      DEPENDS
      HIOP_SPARSE
      COMMAND
      ${RUNCMD}
      $<TARGET_FILE:test_eqjac_compare>
      -netfile
      ${prefix_net}testx3.m
    )
  endif()
endif()

if(EXAGO_RUN_TESTS)
  foreach(
    model
    solver
    dependency
    IN
    ZIP_LISTS
    models
    solvers
    dependencies
  )
    # Iterate over networks, matching network file to num_copies
    foreach(num IN LISTS num_copies)
      set(testname
          "UNIT_TESTS_EQUALITY_CONSTRAINT_JACOBIAN_${num}_${solver}_${model}"
      )
      exago_add_test(
        NAME
        ${testname}
        DEPENDS
        ${dependency}
        COMMAND
        ${RUNCMD}
        $<TARGET_FILE:jac_eq_acopf>
        -opflow_model
        ${model}
        -opflow_solver
        ${solver}
        -num_copies
        ${num}
      )
      set_tests_properties(${testname} PROPERTIES SKIP_RETURN_CODE 2)
    endforeach()
  endforeach()
endif()
