cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(PoolGameTracking)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRs}
                    include)

add_library(utils
            include/utils.h
            src/utils.cpp)

add_executable(main src/main.cpp)

target_link_libraries(main 
                    ${OpenCV_LIBS}
                    utils)

# DEBUG
add_executable(mask_campo src/mask\ campo\ e\ palline\ _\ bounding\ box\ palline.cpp)
target_link_libraries(mask_campo 
                    ${OpenCV_LIBS})

