cmake_minimum_required(VERSION 3.29)

project(fft)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)  # Use -std=c23 instead of -std=gnu23
set(CMAKE_C_VISIBILITY_PRESET hidden)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Use -std=c++23 instead of -std=gnu++23
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# option(USE_LTO "Enable Link-Time Optimization for Release builds" ON)
# option(USE_LIBC++ "Add the compilation flag -stdlib=libc++" OFF)
# option(BUILD_TESTS "Build the tests" ON)

# https://www.kitware.com/cmake-and-the-default-build-type/
# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
# https://github.com/ninja-build/ninja/issues/174
if(CMAKE_GENERATOR STREQUAL "Ninja")
  add_compile_options($<$<COMPILE_LANG_AND_ID:C,GNU>:$<$<VERSION_GREATER_EQUAL:$<C_COMPILER_VERSION>,4.9>:-fdiagnostics-color=always>>)
  add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,GNU>:$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,4.9>:-fdiagnostics-color=always>>)
  add_compile_options($<$<COMPILE_LANG_AND_ID:C,Clang>:$<$<VERSION_GREATER_EQUAL:$<C_COMPILER_VERSION>,3.5>:-fcolor-diagnostics>>)
  add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,Clang>:$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,3.5>:-fcolor-diagnostics>>)
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(MSVC)
  add_compile_options(/utf-8 /W4 /WX)
  add_compile_options($<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,19.10>:/permissive->)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(fft main.cpp)
