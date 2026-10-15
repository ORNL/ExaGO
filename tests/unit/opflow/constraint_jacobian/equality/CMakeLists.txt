if(EXAGO_ENABLE_RAJA AND EXAGO_ENABLE_CUDA)
  set_source_files_properties(jac_eq_acopf.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(jac_eq_acopf jac_eq_acopf.cpp)
target_link_libraries(jac_eq_acopf ExaGO::OPFLOW)
target_include_directories(
  jac_eq_acopf PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/utils
)

# Numbers of copies of the base network to test
set(num_copies 1 3 600)

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
if(EXAGO_ENABLE_HIOP_SPARSE)
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
    # Iterate over the numbers of copies
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
        -netfile
        ${CMAKE_CURRENT_SOURCE_DIR}/CECJ_unittestx${num}.m
        -num_copies
        ${num}
        -validation
        ${CMAKE_CURRENT_SOURCE_DIR}/cecj.csv
      )
      set_tests_properties(${testname} PROPERTIES SKIP_RETURN_CODE 2)
    endforeach()
  endforeach()
endif()
