set(VENV_PYTHON "${Python_EXECUTABLE}")
set(INSTALL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(INSTALL_CODE "${CMAKE_CURRENT_SOURCE_DIR}")
set(INSTALL_DATA "${CMAKE_CURRENT_SOURCE_DIR}/data")
configure_file(
  server.py.in
  server.py
  @ONLY
  FILE_PERMISSIONS
  OWNER_READ
  OWNER_WRITE
  OWNER_EXECUTE
  GROUP_READ
  GROUP_EXECUTE
  WORLD_READ
  WORLD_EXECUTE
)

install(
  DIRECTORY backend
  DESTINATION "${EXAGO_INSTALL_LIBEXECDIR}/viz"
  PATTERN "config.py" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
install(DIRECTORY assets DESTINATION "${EXAGO_INSTALL_DATADIR}/viz")
exago_install_python_app(viz "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt")

install(
  CODE [=[
set(VENV_PYTHON "${EXAGO_VENV_DIR}/bin/python")
set(INSTALL_ROOT "${EXAGO_APP_DATA_DIR}")
set(INSTALL_CODE "${EXAGO_APP_DIR}")
set(INSTALL_DATA "${EXAGO_DATADIR}/datafiles/viz")
exago_configure_install("${EXAGO_PYTHON_APP_SOURCE}/server.py.in" "${EXAGO_APP_DIR}/server.py")
exago_install_launcher("run.sh.in" "exago_viz")
]=]
)
