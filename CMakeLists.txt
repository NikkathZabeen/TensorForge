cmake_minimum_required(VERSION 3.20)
project(TensorJIT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)

if(LLVM_VERSION_MAJOR LESS 17)
    message(FATAL_ERROR "TensorJIT requires LLVM 17 or newer; found LLVM ${LLVM_PACKAGE_VERSION}")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_compile_definitions(${LLVM_DEFINITIONS_LIST})

llvm_map_components_to_libnames(llvm_libs
    support
    core
    irreader
    orcjit
    native
    nativecodegen
    passes
    scalaropts
    instcombine
)

add_library(tensorjit_core
    src/lexer.cpp
    src/parser.cpp
    src/codegen.cpp
    src/optimizer.cpp
    src/jit.cpp
)
target_link_libraries(tensorjit_core PUBLIC ${llvm_libs})

add_executable(tensoriit main.cpp)
target_link_libraries(tensoriit PRIVATE tensorjit_core)

add_executable(test_lexer tests/test_lexer.cpp)
target_link_libraries(test_lexer PRIVATE tensorjit_core)

add_executable(test_parser tests/test_parser.cpp)
target_link_libraries(test_parser PRIVATE tensorjit_core)

add_executable(test_codegen tests/test_codegen.cpp)
target_link_libraries(test_codegen PRIVATE tensorjit_core)

add_executable(bench_matmul benchmarks/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE tensorjit_core)

enable_testing()
add_test(NAME lexer COMMAND test_lexer)
add_test(NAME parser COMMAND test_parser)
add_test(NAME codegen COMMAND test_codegen)
