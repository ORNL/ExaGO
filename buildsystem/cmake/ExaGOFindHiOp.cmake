#[[

Calls find_package to search for HiOp config. Requires HiOp >= 0.5.3

User may set:
- HiOp_DIR
- HiOp_ROOT

]]

find_package(HiOp REQUIRED)

# HiOp's exported targets reference the imported target STRUMPACK::strumpack
# when HiOp is built with the STRUMPACK backend, but HiOpConfig.cmake does not
# find_package(STRUMPACK) itself, so the consumer has to. Without this, the
# configure fails in HiOpTargets.cmake with
#   The link interface of target "HiOp::STRUMPACK" contains: STRUMPACK::strumpack
#   but the target was not found.
if(NOT TARGET STRUMPACK::strumpack AND DEFINED HiOp_DIR)
  file(STRINGS "${HiOp_DIR}/HiOpTargets.cmake" _exago_hiop_strumpack_refs
       REGEX "STRUMPACK::strumpack")
  if(_exago_hiop_strumpack_refs)
    find_package(strumpack REQUIRED PATHS ${STRUMPACK_DIR} NO_MODULE)
    message(STATUS "Loaded STRUMPACK (referenced by HiOp's targets) from ${strumpack_DIR}")
  endif()
endif()

if(TARGET HiOp::HiOp)
  if(HiOp::SPARSE AND TARGET HiOp::COINHSL)
    set(EXAGO_ENABLE_HIOP_SPARSE ON)
  endif()
  mark_as_advanced(FORCE HiOp::SPARSE)
else()
  message(FATAL_ERROR "Find_package could not load HiOp")
endif()
