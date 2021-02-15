cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mppi-tensorflow LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

#################
# ADD LIBRARIES #
#################

set(USE_GL 1)

#link to cuda if it's available
find_package(CUDA)
find_package(TensorflowCC REQUIRED)

include_directories(include include/mujoco)

# N.B. This is a temporary flag. It only really applies to Linux, as Mac
# does not need X11.
option(RUN_X11_TESTS "Run tests that require X11" OFF)

set(MUJOCO_MJPRO_PATH ${CMAKE_CURRENT_SOURCE_DIR})

link_directories(lib lib/contrib/lib/)
message(STATUS "MuJoCo path: " ${MUJOCO_MJPRO_PATH})

#Finding main mujoco library
if(${USE_GL})
    file(GLOB LIB_MUJOCO ${MUJOCO_MJPRO_PATH}/lib/contrib/lib/libmujoco[0-9][0-9][0-9].so)
else()
    file(GLOB LIB_MUJOCO ${MUJOCO_MJPRO_PATH}/lib/contrib/lib/libmujoco[0-9][0-9][0-9]nogl.so)
endif()

#Showing mujoco library found
message(STATUS "MuJoCo lib: " ${LIB_MUJOCO})

find_package(glfw3 3.4 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRECTORIES})
message(STATUS "GLFW path: " ${GLFW_INCULUDE_DIRECTORIES})

include_directories(${GLEW_INCULUDE_DIRECTORIES})
message(STATUS "GLEW path: " ${GLEW_INCULUDE_DIRECTORIES})

find_package(GLEW 2.0 REQUIRED)
find_package(PkgConfig)

pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)
include_directories(${YAMLCPP_INCLUDE_DIRS})

######################
# EXECUTABLE SECTION #
######################

set(BIN mppi-tf)

add_executable(${BIN}
                src/main.cpp
                src/utile.cpp
                src/controller_base.cpp
                src/cost_base.cpp
                src/model_base.cpp
                src/data_base.cpp
                src/mj_pm_env.cpp
                src/mj_env.cpp)


target_link_libraries(${BIN} TensorflowCC::TensorflowCC)
target_link_libraries(${BIN} ${LIB_MUJOCO})
target_link_libraries(${BIN} GL GLU glut)
target_link_libraries(${BIN} ${GLEW_LIBRARIES} glfw)
target_link_libraries(${BIN} ${catkin_LIBRARIES}  ${YAMLCPP_LIBRARIES})

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
                test/test_controller.cpp src/controller_base.cpp
                src/data_base.cpp)

add_test(NAME ${TEST_BIN} COMMAND ${TEST_BIN})
target_link_libraries(${TEST_BIN} PUBLIC gtest gtest_main pthread TensorflowCC::TensorflowCC)
