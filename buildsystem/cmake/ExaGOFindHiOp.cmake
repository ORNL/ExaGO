#[[

Calls find_package to search for HiOp config. Requires HiOp >= 0.5.3

User may set:
- HiOp_DIR
- HiOp_ROOT

]]

find_package(HiOp REQUIRED)

if(TARGET HiOp::HiOp)
  # HiOp's sparse interface does not require COINHSL -- MA57 is only one of its
  # backends (STRUMPACK, cuSOLVER-LU/ReSolve and cuDSS are others). Gating on
  # HiOp::COINHSL leaves PBPOLRAJAHIOPSPARSE unregistered for a perfectly
  # functional HiOp built with a free backend, and a run then fails with
  # "Unknown type for OPFLOW Model PBPOLRAJAHIOPSPARSE".
  if(HiOp::SPARSE)
    set(EXAGO_ENABLE_HIOP_SPARSE ON)
  endif()
  mark_as_advanced(FORCE HiOp::SPARSE)
else()
  message(FATAL_ERROR "Find_package could not load HiOp")
endif()
