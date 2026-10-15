set(app_dest "${EXAGO_INSTALL_LIBEXECDIR}/agentigrid")
install(
  DIRECTORY agentigrid launcher tests
  DESTINATION "${app_dest}"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
)
install(FILES pytest.ini DESTINATION "${app_dest}")
install(
  DIRECTORY configs
  DESTINATION "${EXAGO_INSTALL_DATADIR}/agentigrid"
  PATTERN "env_setup.sh" EXCLUDE
  PATTERN "local_config.yaml" EXCLUDE
  PATTERN "logs" EXCLUDE
  PATTERN "*.in" EXCLUDE
)
install(FILES README.md agentigrid_architecture.md
        DESTINATION "${CMAKE_INSTALL_DOCDIR}/agentigrid"
)
install(FILES applications/gridkit/README.md
        DESTINATION "${app_dest}/applications/gridkit"
)
install(FILES data/gridkit/README.md
        DESTINATION "${EXAGO_INSTALL_DATADIR}/datafiles/gridkit"
)
exago_install_python_app(
  agentigrid "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/launcher/requirements.txt"
)

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
    agentigrid_pytest
    PROPERTIES LABELS "agentigrid;python" TIMEOUT 300 ENVIRONMENT
               "EXAGO_TEST_BINARY_DIR=${PROJECT_BINARY_DIR}/applications"
  )
endif()

install(
  CODE [=[
set(AGENTIGRID_BINARY_DIR "${EXAGO_BINDIR}")
set(AGENTIGRID_DATA_DIR "${EXAGO_DATADIR}/datafiles")
exago_configure_install("${EXAGO_PYTHON_APP_SOURCE}/configs/installed_config.yaml.in"
                       "${EXAGO_APP_DATA_DIR}/configs/default_config.yaml")
exago_install_launcher("agentigrid.sh.in" "agentigrid")
exago_install_launcher("agentigrid_launcher.sh.in" "agentigrid_launcher")

set(examples_dir "${EXAGO_APP_DIR}/data/exago/examples")
file(MAKE_DIRECTORY "$ENV{DESTDIR}${examples_dir}")
file(GLOB examples "${EXAGO_PYTHON_APP_SOURCE}/data/exago/examples/*")
foreach(example IN LISTS examples)
  get_filename_component(alias "${example}" NAME)
  get_filename_component(source "${example}" REALPATH)
  get_filename_component(name "${source}" NAME)
  if(source MATCHES "/case9/")
    set(case_dir "case9")
  elseif(name STREQUAL "ieee_118_bus_v10.m" OR name MATCHES "^case118")
    set(case_dir "case118")
  elseif(name MATCHES "ACTIVSg(2000|200|500|10k)")
    set(case_dir "case_ACTIVSg${CMAKE_MATCH_1}")
  else()
    get_filename_component(case_dir "${source}" NAME_WE)
  endif()
  file(RELATIVE_PATH target "${examples_dir}" "${AGENTIGRID_DATA_DIR}/${case_dir}/${name}")
  file(CREATE_LINK "${target}" "$ENV{DESTDIR}${examples_dir}/${alias}" SYMBOLIC)
  list(APPEND CMAKE_INSTALL_MANIFEST_FILES "${examples_dir}/${alias}")
endforeach()
]=]
)
