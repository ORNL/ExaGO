set(data_dest "${EXAGO_INSTALL_DATADIR}/datafiles")
install(FILES "${PROJECT_SOURCE_DIR}/tests/data/ieee_118_bus_v10.m"
        DESTINATION "${data_dest}/case118"
)

file(GLOB data_entries "${PROJECT_SOURCE_DIR}/datafiles/*")
foreach(entry IN LISTS data_entries)
  get_filename_component(name "${entry}" NAME)
  if(IS_DIRECTORY "${entry}")
    if(name STREQUAL "TAMU200_scenarios")
      install(DIRECTORY "${entry}" DESTINATION "${data_dest}/case_ACTIVSg200")
    else()
      install(DIRECTORY "${entry}" DESTINATION "${data_dest}")
    endif()
  else()
    if(name MATCHES "ACTIVSg(2000|200|500|10k)")
      set(case_dir "case_ACTIVSg${CMAKE_MATCH_1}")
    elseif(name MATCHES "^case118")
      set(case_dir "case118")
    else()
      get_filename_component(case_dir "${entry}" NAME_WE)
    endif()
    install(FILES "${entry}" DESTINATION "${data_dest}/${case_dir}")
  endif()
endforeach()

file(GLOB viz_inputs "${PROJECT_SOURCE_DIR}/viz/data/*.gic")
foreach(entry IN LISTS viz_inputs)
  get_filename_component(name "${entry}" NAME)
  if(name MATCHES "ACTIVSg(2000|200)")
    install(FILES "${entry}"
            DESTINATION "${data_dest}/case_ACTIVSg${CMAKE_MATCH_1}"
    )
  endif()
endforeach()
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/viz/data/"
  DESTINATION "${data_dest}/viz"
  FILES_MATCHING
  PATTERN "*.json"
  PATTERN "*.csv"
)
