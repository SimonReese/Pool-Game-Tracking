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


# Add classes --------------------------------------
# Evaluation metrics class
add_library(EvaluationMetrics
            include/EvaluationMetrics.h
            src/EvaluationMetrics.cpp)

# Drawing class
add_library(DrawingClass
            include/Draw.h
            src/Draw.cpp)

# TableSegmenter class
add_library(TableSegmenter
            include/TableSegmenter.h
            src/TableSegmenter.cpp)

# BallTracker class
add_library(BallDetector
            include/BallDetector.h
            src/BallDetector.cpp)

# BallClassifier class
add_library(BallClassifier
            include/BallClassifier.h
            src/BallClassifier.cpp)

# BallTracker class
add_library(BallTracker 
            include/BallTracker.h
            src/BallTracker.cpp)

# Ball class
add_library(Ball
            include/Ball.h
            src/Ball.cpp)

# Add executables -----------------------------------
add_executable(evaluation_runner src/evaluation-runner.cpp)
add_executable(draw_class src/draw-class.cpp)
add_executable(open_mask src/open-mask.cpp)


# Link libraries to files
target_link_libraries(evaluation_runner ${OpenCV_LIBS} EvaluationMetrics TableSegmenter BallDetector BallClassifier Ball)
target_link_libraries(draw_class ${OpenCV_LIBS} DrawingClass TableSegmenter BallClassifier BallDetector BallTracker Ball)
target_link_libraries(open_mask ${OpenCV_LIBS})

target_link_libraries(EvaluationMetrics ${OpenCV_LIBS})
target_link_libraries(DrawingClass ${OpenCV_LIBS} Ball)
target_link_libraries(TableSegmenter ${OpenCV_LIBS})
target_link_libraries(BallClassifier ${OpenCV_LIBS})
target_link_libraries(BallDetector ${OpenCV_LIBS} Ball)
target_link_libraries(BallTracker ${OpenCV_LIBS} Ball) 

target_link_libraries(Ball ${OpenCV_LIBS})

