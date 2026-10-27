cmake_minimum_required(VERSION 3.20)
project(bpe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BPE_BUILD_TESTS "Build unit tests" ON)
option(BPE_BUILD_PYTHON "Build pybind11 module" ON)
option(BPE_NATIVE "Add -march=native" OFF)
option(BPE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(bpe
  src/encoding.cpp
  src/merge.cpp
  src/load.cpp
  src/trainer.cpp
)
target_include_directories(bpe
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(bpe PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(bpe PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BPE_NATIVE)
  target_compile_options(bpe PRIVATE -march=native)
endif()

if(BPE_SANITIZERS)
  target_compile_options(bpe PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(bpe PUBLIC -fsanitize=address,undefined)
endif()

function(bpe_apply_sanitizers target)
  if(BPE_SANITIZERS)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()

include(CheckIPOSupported)
check_ipo_supported(RESULT bpe_ipo_ok OUTPUT bpe_ipo_msg)
# The static library is bitcode under Release IPO. Dependents must enable IPO
# too, or clang's link against GNU ld rejects the archive.
function(bpe_enable_ipo target)
  if(bpe_ipo_ok)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endfunction()
bpe_enable_ipo(bpe)

add_executable(bpe_cli src/cli.cpp)
target_link_libraries(bpe_cli PRIVATE bpe)
target_include_directories(bpe_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(bpe_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(bpe_cli PROPERTIES OUTPUT_NAME bpe)
if(BPE_NATIVE)
  target_compile_options(bpe_cli PRIVATE -march=native)
endif()
bpe_apply_sanitizers(bpe_cli)
bpe_enable_ipo(bpe_cli)

if(BPE_BUILD_TESTS)
  include(FetchContent)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(bpe_tests
    tests/test_core.cpp
    tests/test_tiktoken.cpp
  )
  target_link_libraries(bpe_tests PRIVATE bpe GTest::gtest_main)
  target_include_directories(bpe_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(bpe_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_definitions(bpe_tests PRIVATE
    BPE_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    BPE_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
  )
  if(BPE_SANITIZERS)
    target_compile_definitions(bpe_tests PRIVATE BPE_SANITIZERS)
  endif()
  bpe_apply_sanitizers(bpe_tests)
  bpe_enable_ipo(bpe_tests)
  include(GoogleTest)
  gtest_discover_tests(bpe_tests)
endif()

if(BPE_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
  if(Python3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(pybind11)
    pybind11_add_module(bpe_py python/bindings.cpp)
    target_link_libraries(bpe_py PRIVATE bpe)
    target_compile_options(bpe_py PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(bpe_py PROPERTIES
      OUTPUT_NAME bpe
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
    if(BPE_NATIVE)
      target_compile_options(bpe_py PRIVATE -march=native)
    endif()
    bpe_enable_ipo(bpe_py)
  else()
    message(STATUS "Python development headers not found; skipping bindings")
  endif()
endif()
