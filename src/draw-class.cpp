/**
 * @author Simone Peraro.
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Draw.h"

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
    // Show
    cv::imshow("From drawing class", result);
    cv::waitKey(0);
    return 0;
}