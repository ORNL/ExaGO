# AgentiGrid — CMake integration with ExaGO
#
# AgentiGrid is a Python subproject that uses large language models as
# autonomous iterative decision-making agents driving ExaGO simulations. This
# CMakeLists.txt: * Installs AgentiGrid sources to
# ${CMAKE_INSTALL_PREFIX}/share/exago/agentigrid/ * Optionally runs `pip install
# -e .` at install time (best-effort) * Registers a CTest entry for AgentiGrid's
# fast unit tests
#
# This file is invoked from the root CMakeLists.txt via
# add_subdirectory(agentigrid), inside an if(EXAGO_ENABLE_AGENTIGRID) block.

set(__install_dest share/exago/agentigrid)

# ── Source installation ────────────────────────────────────────────────────

# Install the inner Python package (agentigrid/agentigrid/)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/agentigrid
  DESTINATION ${__install_dest}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Install the Streamlit launcher
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/launcher
  DESTINATION ${__install_dest}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Install the test suite
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
  DESTINATION ${__install_dest}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
)

# Install configs, but exclude machine-specific instantiations
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/configs
  DESTINATION ${__install_dest}
  PATTERN "env_setup.sh" EXCLUDE
  PATTERN "local_config.yaml" EXCLUDE
  PATTERN "logs" EXCLUDE
)

# Install top-level project files
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/pytest.ini
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        ${CMAKE_CURRENT_SOURCE_DIR}/agentigrid_architecture.md
        ${CMAKE_CURRENT_SOURCE_DIR}/run_agentigrid.sh
  DESTINATION ${__install_dest}
)

# Install README placeholders for applications/ and data/ subdirs. The binaries
# and data files themselves are not shipped — AgentiGrid uses ExaGO's canonical
# binaries (${CMAKE_INSTALL_PREFIX}/bin/) and datafiles
# (${CMAKE_INSTALL_PREFIX}/share/exago/datafiles/) at runtime. The READMEs
# document this convention.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/applications/README.md
        DESTINATION ${__install_dest}/applications
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/README.md
        DESTINATION ${__install_dest}/data
)

# ── Install-time pip install ───────────────────────────────────────────────

# Run `pip install -e .` at install time to register AgentiGrid with the active
# Python environment. This is best-effort: a failure here does not abort the
# ExaGO install. The user can recover by running `pip install -e .` manually
# from the install directory.
string(
  JOIN
  "\n"
  install_script
  "set(Python_EXECUTABLE \"${Python_EXECUTABLE}\")"
  [[
  set(INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/share/exago/agentigrid)
  message(STATUS "Installing AgentiGrid Python package from ${INSTALL_DIR}")
  execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pip install -e .
      WORKING_DIRECTORY "${INSTALL_DIR}"
      RESULT_VARIABLE PIP_RESULT
  )
  if(NOT PIP_RESULT EQUAL 0)
      message(WARNING
          "pip install -e . failed in ${INSTALL_DIR} (exit code: ${PIP_RESULT}). "
          "AgentiGrid sources have been installed, but the 'agentigrid' command "
          "was not registered with your Python environment. "
          "To complete the installation manually, run: "
          "cd ${INSTALL_DIR} && pip install -e ."
      )
  else()
      message(STATUS "AgentiGrid Python package installed successfully")
  endif()
  ]]
)
install(CODE "${install_script}")

# ── CTest integration ──────────────────────────────────────────────────────

# Register a single CTest entry that runs AgentiGrid's fast unit tests. Slow
# tests (marked @pytest.mark.slow in pytest.ini) are excluded — they require LLM
# API keys and network access.
if(EXAGO_RUN_TESTS)
  add_test(
    NAME agentigrid_pytest
    COMMAND "${Python_EXECUTABLE}" -m pytest -m "not slow" --tb=short
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(
    agentigrid_pytest PROPERTIES LABELS "agentigrid;python" TIMEOUT 300
  )
endif()
