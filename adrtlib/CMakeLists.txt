cmake_minimum_required(VERSION 3.15...3.27)
project(adrtlib)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (CMAKE_BUILD_TYPE)
  message(WARNING "CMAKE_BUILD_TYPE must not be specified."
          " And someone set this variable to '${CMAKE_BUILD_TYPE}'")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_adrtlib NOMINSIZE _adrtlib.cpp)
target_include_directories(_adrtlib PRIVATE include)
target_compile_features(_adrtlib PRIVATE cxx_std_17)

install(TARGETS _adrtlib LIBRARY DESTINATION adrtlib)

#
# testing
#

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  adrtlib_test
  test/adrtlib_test.cpp
)
target_include_directories(adrtlib_test PRIVATE include)
target_compile_features(adrtlib_test PRIVATE cxx_std_17)

target_link_libraries(
  adrtlib_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(adrtlib_test)

#
# benchmarking
#

FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.zip
)

FetchContent_MakeAvailable(googlebenchmark)


add_executable(
  adrtlib_benchmark
  benchmark/adrtlib_benchmark.cpp
)
target_include_directories(adrtlib_benchmark PRIVATE include)
target_compile_features(adrtlib_benchmark PRIVATE cxx_std_17)

target_link_libraries(
  adrtlib_benchmark
  benchmark::benchmark
)
