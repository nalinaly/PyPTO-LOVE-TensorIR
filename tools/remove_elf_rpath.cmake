cmake_minimum_required(VERSION 3.31)

if(NOT DEFINED PYPTO_ELF_PATH OR PYPTO_ELF_PATH STREQUAL "")
  message(FATAL_ERROR "PYPTO_ELF_PATH is required")
endif()
if(NOT IS_ABSOLUTE "${PYPTO_ELF_PATH}")
  message(FATAL_ERROR "PYPTO_ELF_PATH must be absolute")
endif()
if(NOT EXISTS "${PYPTO_ELF_PATH}" OR IS_DIRECTORY "${PYPTO_ELF_PATH}")
  message(FATAL_ERROR "PYPTO_ELF_PATH must name an existing regular file")
endif()
if(IS_SYMLINK "${PYPTO_ELF_PATH}")
  message(FATAL_ERROR "PYPTO_ELF_PATH must not be a symlink")
endif()

get_filename_component(pypto_elf_real "${PYPTO_ELF_PATH}" REALPATH)
if(NOT pypto_elf_real STREQUAL PYPTO_ELF_PATH)
  message(FATAL_ERROR "PYPTO_ELF_PATH must be canonical")
endif()

# CMake's ELF editor removes DT_RPATH/DT_RUNPATH without introducing an
# additional host utility into the reviewed producer closure.
file(RPATH_REMOVE FILE "${PYPTO_ELF_PATH}")
