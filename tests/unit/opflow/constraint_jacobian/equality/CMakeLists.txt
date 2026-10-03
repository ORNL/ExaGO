if(EXAGO_ENABLE_RAJA AND EXAGO_ENABLE_CUDA)
  set_source_files_properties(jac_eq_acopf.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(jac_eq_acopf jac_eq_acopf.cpp)
target_link_libraries(jac_eq_acopf ExaGO::OPFLOW)
target_include_directories(
  jac_eq_acopf PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/utils
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
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/CECJ_unittestx${n}.m
    ${CMAKE_CURRENT_BINARY_DIR}/CECJ_unittestx${n}.m COPYONLY
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
          "UNIT_TESTS_OPFLOW_EQUALITY_CONSTRAINT_JACOBIAN_${num}_${solver}_${model}"
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
