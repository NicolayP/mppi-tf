cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(test_app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test_app run_model.cpp)
target_link_libraries(test_app "${TORCH_LIBRARIES}")
set_property(TARGET test_app PROPERTY CXX_STANDARD 17)

