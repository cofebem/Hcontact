cmake_minimum_required(VERSION 3.18)
project(hmatrix_contact CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Let an activated conda env (fenicsx-env) provide Eigen and Python.
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

find_package(Eigen3 3.4 REQUIRED)
find_package(OpenMP)

add_library(hmatrix_contact_core STATIC
  src/boussinesq_kernel.cpp
  src/cluster_tree.cpp
)
target_include_directories(hmatrix_contact_core PUBLIC include)
target_link_libraries(hmatrix_contact_core PUBLIC Eigen3::Eigen)
target_compile_options(hmatrix_contact_core PRIVATE -Wall -Wextra)
set_target_properties(hmatrix_contact_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
  target_link_libraries(hmatrix_contact_core PUBLIC OpenMP::OpenMP_CXX)
endif()

enable_testing()
foreach(t kernel hmatrix)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE hmatrix_contact_core)
  target_compile_options(test_${t} PRIVATE -Wall -Wextra)
  add_test(NAME ${t} COMMAND test_${t})
endforeach()

# pybind11 is not installed in fenicsx-env; pass its cmake dir explicitly, e.g.
#   -Dpybind11_DIR=$(conda run -n dolfinx-010 python -m pybind11 --cmakedir)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/python/bindings.cpp)
  pybind11_add_module(hmatrix_contact python/bindings.cpp)
  target_link_libraries(hmatrix_contact PRIVATE hmatrix_contact_core)
  set_target_properties(hmatrix_contact PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
endif()
