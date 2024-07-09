cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
project(PoolGameTracking)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS}
                    include)

add_library(utils
            include/utils.h
            src/utils.cpp)

add_executable(main src/main.cpp)

target_link_libraries(main 
                    ${OpenCV_LIBS}
                    utils)

# Evaluation metrics
add_library(EvaluationMetrics
            include/EvaluationMetrics.h
            src/EvaluationMetrics.cpp)

add_executable(evaluation_runner src/evaluation-runner.cpp)

target_link_libraries(evaluation_runner ${OpenCV_LIBS} EvaluationMetrics)
target_link_libraries(EvaluationMetrics ${OpenCV_LIBS})


# Drawing class
add_library(DrawingClass
            include/Draw.h
            src/Draw.cpp)

add_executable(draw_class src/draw-class.cpp)

target_link_libraries(DrawingClass ${OpenCV_LIBS})
target_link_libraries(draw_class ${OpenCV_LIBS} DrawingClass)

# DEBUG
add_library(MaskAndBalls
            include/BallsDetection.h
            include/FieldGeometryAndMask.h
            include/Ball.h
            src/BallsDetection.cpp
            src/FieldGeometryAndMask.cpp
            src/Ball.cpp)

add_executable(mask_campo src/mask_campo_e_palline_bounding_box_palline.cpp)
target_link_libraries(mask_campo 
                    ${OpenCV_LIBS}
                    MaskAndBalls)
                    target_link_libraries(MaskAndBalls ${OpenCV_LIBS})

