# Setup project requirements
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
project(PoolGameTracking)

# Required libraries
find_package(OpenCV REQUIRED)

# Select directories for including headers
include_directories(${OPENCV_INCLUDE_DIRS}
                    include)


# Add libraries ------------------------------------
# Utility library - TODO: check if needed
add_library(utils
            include/utils.h
            src/utils.cpp)

# DEBUG
add_library(MaskAndBalls
            include/BallsDetection.h
            include/FieldGeometryAndMask.h
            include/Ball.h
            src/BallsDetection.cpp
            src/FieldGeometryAndMask.cpp
            src/Ball.cpp)

# Add classes --------------------------------------
# Evaluation metrics class
add_library(EvaluationMetrics
            include/EvaluationMetrics.h
            src/EvaluationMetrics.cpp)

# Drawing class
add_library(DrawingClass
            include/Draw.h
            src/Draw.cpp)

# Tracking class
add_library(TrackingClass
            include/MyTracking.h
            src/MyTracking.cpp)


# Add executables -----------------------------------
add_executable(main src/main.cpp)
add_executable(evaluation_runner src/evaluation-runner.cpp)
add_executable(draw_class src/draw-class.cpp)
add_executable(mask_campo src/mask_campo_e_palline_bounding_box_palline.cpp)

# Link libraries to files
target_link_libraries(main ${OpenCV_LIBS} utils)
target_link_libraries(evaluation_runner ${OpenCV_LIBS} EvaluationMetrics)
target_link_libraries(draw_class ${OpenCV_LIBS} DrawingClass)
target_link_libraries(mask_campo ${OpenCV_LIBS} MaskAndBalls TrackingClass)

target_link_libraries(EvaluationMetrics ${OpenCV_LIBS} utils)
target_link_libraries(DrawingClass ${OpenCV_LIBS})
target_link_libraries(MaskAndBalls ${OpenCV_LIBS})
target_link_libraries(TrackingClass ${OpenCV_LIBS})

