set(EXAGO_INSTALL_LIBEXECDIR "${CMAKE_INSTALL_LIBEXECDIR}/exago")
set(EXAGO_INSTALL_DATADIR "${CMAKE_INSTALL_DATADIR}/exago")
option(
  EXAGO_INSTALL_PYTHON_DEPENDENCIES
  "Install Python application dependencies into private virtual environments"
  ON
)

function(exago_install_python_app app)
  set(EXAGO_PYTHON_APP "${app}")
  set(EXAGO_PYTHON_APP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}")
  set(EXAGO_PYTHON_APP_REQUIREMENTS ${ARGN})
  configure_file(
    "${PROJECT_SOURCE_DIR}/buildsystem/cmake/ExaGOInstallPythonApp.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/install_python_app.cmake" @ONLY
  )
  install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/install_python_app.cmake")
endfunction()
