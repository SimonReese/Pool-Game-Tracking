/**
 * @author Simone Peraro.
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Draw.h"

#include <opencv2/imgproc.hpp>

int main(int argc, char* argv[]){
    if (argc < 2){
        std::cerr << "Please provide a video or an image path." << std::endl;
        return -1;
    }

    std::string inputFile = argv[1];

    // Read a single frame image
    cv::Mat frame = cv::imread(inputFile);
    // Construct Draw object
    Draw drawing("");
    // Set frame
    drawing.setCurrentFrame(frame);
    // Get drawing
    cv::Mat result;
    drawing.getGameDraw(result);
    cv::imshow("Correction", result);
    cv::imshow("Original", frame);

    // Testing overlapping function
    cv::Mat spiderMan = cv::imread("../res/spider-man-3.png");
    cv::Mat testImage = cv::imread("../res/opencv-test-image.png");
    cv::resize(spiderMan, spiderMan, cv::Size(300, 300));
    cv::Mat over = drawing.drawOver(testImage, spiderMan, cv::Point(testImage.cols / 2, testImage.rows / 2));
    cv::imshow("Overlap", over);
    //animation();
    cv::waitKey(0);
    return 0;
}