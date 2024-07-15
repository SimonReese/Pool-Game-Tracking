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
add_library(FilesystemUtils
            include/FilesystemUtils.h
            src/FilesystemUtils.cpp)

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
target_link_libraries(BallTracker ${OpenCV_LIBS})       



# Add executables -----------------------------------
add_executable(main src/main.cpp)
add_executable(evaluation_runner src/evaluation-runner.cpp)
add_executable(draw_class src/draw-class.cpp)
add_executable(mask_campo src/mask_campo_e_palline_bounding_box_palline.cpp)

#just for testing purposes
add_executable(testTracker src/testTracker.cpp)
target_link_libraries(testTracker ${OpenCV_LIBS} MaskAndBalls BallTracker)


# Link libraries to files
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(evaluation_runner ${OpenCV_LIBS} EvaluationMetrics)
target_link_libraries(draw_class ${OpenCV_LIBS} DrawingClass TableSegmenter BallClassifier FilesystemUtils BallDetector)
target_link_libraries(mask_campo ${OpenCV_LIBS} MaskAndBalls BallClassifier FilesystemUtils)

target_link_libraries(EvaluationMetrics ${OpenCV_LIBS} FilesystemUtils)
target_link_libraries(DrawingClass ${OpenCV_LIBS} MaskAndBalls)
target_link_libraries(MaskAndBalls ${OpenCV_LIBS})
target_link_libraries(TableSegmenter ${OpenCV_LIBS})
target_link_libraries(BallClassifier ${OpenCV_LIBS})
target_link_libraries(BallDetector ${OpenCV_LIBS} MaskAndBalls)

target_link_libraries(FilesystemUtils ${OpenCV_LIBS})

