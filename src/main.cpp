#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <utils.h>
std::string PATH_TEST_IMAGE = "../res/opencv-test-image.png";

int main(int argc, char* argv[]){

    helloFunction("World!");
    
    cv::Mat testImage = cv::imread(PATH_TEST_IMAGE);
    cv::imshow("Test Image", testImage);
    cv::waitKey(5);
}