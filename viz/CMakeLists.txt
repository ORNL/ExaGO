set(__install_dest share/exago/viz)

find_program(
  YARN_EXECUTABLE yarn REQUIRED DOC "Path to the yarn package manager"
)
message(STATUS "Found yarn: ${YARN_EXECUTABLE}")

set(YARN_STAMP "${CMAKE_CURRENT_BINARY_DIR}/yarn_install.stamp")
add_custom_command(
  OUTPUT ${YARN_STAMP}
  COMMAND ${YARN_EXECUTABLE} install
  COMMAND ${CMAKE_COMMAND} -E touch ${YARN_STAMP}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Running yarn install in ${CMAKE_CURRENT_SOURCE_DIR}"
  VERBATIM
)
add_custom_target(yarn_install ALL DEPENDS ${YARN_STAMP})

set(EXAGO_VIZ_SUBDIRS backend components data geo_data images src)

set(EXAGO_VIZ_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/exago-logo.png
    ${CMAKE_CURRENT_SOURCE_DIR}/favicon.png
    ${CMAKE_CURRENT_SOURCE_DIR}/index.css
    ${CMAKE_CURRENT_SOURCE_DIR}/index.html
    ${CMAKE_CURRENT_SOURCE_DIR}/package.json
    ${CMAKE_CURRENT_SOURCE_DIR}/tsconfig.json
    ${CMAKE_CURRENT_SOURCE_DIR}/tsconfig.node.json
    ${CMAKE_CURRENT_SOURCE_DIR}/vite.config.ts
)

install(DIRECTORY ${EXAGO_VIZ_SUBDIRS} DESTINATION ${__install_dest})
install(FILES ${EXAGO_VIZ_FILES} DESTINATION ${__install_dest})
string(
  JOIN
  "\n"
  install_script
  "set(YARN_EXECUTABLE ${YARN_EXECUTABLE})"
  [[
  set(INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/share/exago/viz)
  message(STATUS "Running yarn install in ${INSTALL_DIR}")
  execute_process(
      COMMAND ${YARN_EXECUTABLE} install
      WORKING_DIRECTORY ${INSTALL_DIR}
      RESULT_VARIABLE YARN_RESULT
  )
  if(NOT YARN_RESULT EQUAL 0)
      message(FATAL_ERROR
          "yarn install failed in ${INSTALL_DIR} (exit code: ${YARN_RESULT})")
  endif()
  message(STATUS "yarn install complete in ${INSTALL_DIR}")
  ]]
)
install(CODE "${install_script}")
