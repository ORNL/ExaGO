# Generate YAML config and launcher script for without install
set(LAUNCH_EXAGO_DIR ${CMAKE_SOURCE_DIR})
set(LAUNCH_BIN_DIR ${CMAKE_BINARY_DIR}/applications)
set(LAUNCH_EXAGO_PYTHON_PATH ${CMAKE_BINARY_DIR}/interfaces/python)
configure_file(
  ${CMAKE_SOURCE_DIR}/launcher/config.yaml.in
  ${CMAKE_SOURCE_DIR}/launcher/config.yaml @ONLY
)
set(LAUNCH_EXAGO_LAUNCHER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LAUNCH_EXAGO_STREAMLIT_CMD "'${Python_EXECUTABLE} -m streamlit run'")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/run.sh
  @ONLY
)

# Install app and logo
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/app.py
        DESTINATION share/exago/launcher
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/figs/ornl-logo.png
        DESTINATION share/exago/launcher/figs
)

# Delay configuration of installed launcher to install time
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml.in
              ${CMAKE_CURRENT_SOURCE_DIR}/run.sh.in
        DESTINATION share/exago/launcher
)
string(
  JOIN
  "\n"
  install_script
  "set(EXAGO_PYTHON_SITELIB ${EXAGO_PYTHON_SITELIB})"
  "set(LAUNCH_EXAGO_STREAMLIT_CMD \"${LAUNCH_EXAGO_STREAMLIT_CMD}\")"
  [[
  get_filename_component(LAUNCH_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE)
  set(LAUNCH_EXAGO_DIR ${LAUNCH_INSTALL_DIR}/share/exago)
  set(LAUNCH_BIN_DIR ${LAUNCH_INSTALL_DIR}/bin)
  set(LAUNCH_EXAGO_LAUNCHER_DIR ${LAUNCH_EXAGO_DIR}/launcher)
  set(LAUNCH_EXAGO_PYTHON_PATH ${LAUNCH_INSTALL_DIR}/${EXAGO_PYTHON_SITELIB})
  configure_file(
      ${LAUNCH_EXAGO_LAUNCHER_DIR}/config.yaml.in
      ${LAUNCH_EXAGO_LAUNCHER_DIR}/config.yaml @ONLY
  )
  configure_file(
      ${LAUNCH_EXAGO_LAUNCHER_DIR}/run.sh.in
      ${LAUNCH_BIN_DIR}/exago_launcher.sh @ONLY
  )
  file(CHMOD ${LAUNCH_BIN_DIR}/exago_launcher.sh
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ WORLD_READ
  )
  file(REMOVE ${LAUNCH_EXAGO_LAUNCHER_DIR}/config.yaml.in)
  file(REMOVE ${LAUNCH_EXAGO_LAUNCHER_DIR}/run.sh.in)
  ]]
)
install(CODE "${install_script}")
