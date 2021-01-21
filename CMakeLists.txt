cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

######################
# EXECUTABLE SECTION #
######################

project(mppi-tensorflow LANGUAGES CXX)
set(BIN mppi-tf)

include_directories(include)
add_executable(${BIN}
                src/main.cpp
                src/utile.cpp
                src/controller_base.cpp
                src/cost_base.cpp
                src/model_base.cpp)

find_package(TensorflowCC REQUIRED)
target_link_libraries(${BIN} TensorflowCC::TensorflowCC)

#link to cuda if it's available
find_package(CUDA)
if(CUDA_FOUND)
    target_link_libraries(${BIN} ${CUDA_LIBRARIES})
endif()


################
# TEST SECTION #
################

set(TEST_BIN ${BIN}-test)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

add_executable(${TEST_BIN} test/test_entry.cpp
                test/test_utile.cpp src/utile.cpp
                test/test_cost.cpp src/cost_base.cpp
                test/test_model.cpp src/model_base.cpp
                test/test_controller.cpp src/controller_base.cpp)

add_test(NAME ${TEST_BIN} COMMAND ${TEST_BIN})
target_link_libraries(${TEST_BIN} PUBLIC gtest gtest_main pthread TensorflowCC::TensorflowCC)
