cmake_minimum_required(VERSION 3.10)

project("Neural Network"    VERSION 1.0
                            DESCRIPTION "Neural network in C++"
                            LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-Wall")   
ENDIF()

IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "-O3")   
ENDIF()

set(SOURCE_DIR src)

set(SRC_FILES   ${SOURCE_DIR}/main.cpp
                ${SOURCE_DIR}/matrix.cpp
                ${SOURCE_DIR}/matrixfunctions.cpp
                ${SOURCE_DIR}/neuralnetwork.cpp
)


add_executable(neuralnetwork ${SRC_FILES})

set_property(TARGET neuralnetwork PROPERTY CXX_STANDARD 17)
set_property(TARGET neuralnetwork PROPERTY CXX_STANDARD_REQUIRED ON)

