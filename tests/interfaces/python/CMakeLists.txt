if(NOT PY_pytest)
  message(FATAL_ERROR "Pytest module path not found")
endif()

set(pythonpath "$<TARGET_FILE_DIR:exago>:${CMAKE_CURRENT_SOURCE_DIR}:$ENV{PYTHONPATH}")
set(TEST_FILES test_0_initialize.py test_1_pflow.py)
if(EXAGO_ENABLE_HIOP)
  list(APPEND TEST_FILES test_2_opflow.py)
endif()
if(EXAGO_ENABLE_IPOPT)
  list(APPEND TEST_FILES test_3_scopflow.py test_4_sopflow.py test_5_tcopflow.py)
endif()
list(APPEND TEST_FILES test_9_finalize.py)
list(TRANSFORM TEST_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

foreach(marker nocomm MPI)
  if(marker STREQUAL "MPI" AND NOT EXAGO_ENABLE_MPI)
    continue()
  endif()
  if(marker STREQUAL "nocomm")
    set(test_name Python_pytest)
  else()
    set(test_name Python_pytest_MPI)
  endif()
  set(workdir "${CMAKE_CURRENT_BINARY_DIR}/${marker}")
  file(MAKE_DIRECTORY "${workdir}")
  add_test(NAME ${test_name}
           COMMAND ${CMAKE_COMMAND} -E env
                   "PYTHONPATH=${pythonpath}"
                   "EXAGO_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/datafiles"
                   "TMPDIR=${workdir}"
                   ${RUNCMD} "${Python_EXECUTABLE}" -m pytest
                   -m ${marker} -v ${TEST_FILES})
  set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY "${workdir}")
endforeach()
