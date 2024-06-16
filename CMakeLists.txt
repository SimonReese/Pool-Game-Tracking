cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
project(PoolGameTracking)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRs}
                    include)

add_library(utils
            include/utils.h
            src/utils.cpp)

add_library(EvaluationMetrics
            include/EvaluationMetrics.h
            src/EvaluationMetrics.cpp)

add_executable(main src/main.cpp)

target_link_libraries(main 
                    ${OpenCV_LIBS}
                    utils)

# Evaluation metrics
add_executable(evaluation_runner src/evaluation-runner.cpp)

target_link_libraries(evaluation_runner ${OpenCV_LIBS} EvaluationMetrics)
target_link_libraries(EvaluationMetrics ${OpenCV_LIBS})

# DEBUG
add_executable(mask_campo src/mask_campo_e_palline_bounding_box_palline.cpp)
target_link_libraries(mask_campo 
                    ${OpenCV_LIBS})

