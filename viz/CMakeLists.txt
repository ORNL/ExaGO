set(__install_dest share/exago/viz)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(EXAGO_VIZ_SUBDIRS backend data assets)

install(CODE "
  set(VENV_DIR \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${__install_dest}/venv\")
  execute_process(
    COMMAND \"${Python3_EXECUTABLE}\" -m venv \"\${VENV_DIR}\"
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(WIN32)
    set(VENV_PYTHON \"\${VENV_DIR}/Scripts/python.exe\")
  else()
    set(VENV_PYTHON \"\${VENV_DIR}/bin/python\")
  endif()

  set(INSTALL_ROOT \"\${CMAKE_INSTALL_PREFIX}/${__install_dest}\")

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/server.py.in
    \${CMAKE_INSTALL_PREFIX}/${__install_dest}/server.py @ONLY
  )

  execute_process(
    COMMAND \"\${VENV_PYTHON}\" -m pip install
            --ignore-installed
            -r \"${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt\"
    COMMAND_ERROR_IS_FATAL ANY
  )
")
install(DIRECTORY ${EXAGO_VIZ_SUBDIRS} DESTINATION ${__install_dest})
