#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
std::string PATH_TEST_IMAGE = "../res/opencv-test-image.png";

int main(int argc, char* argv[]){
    
    cv::Mat testImage = cv::imread(PATH_TEST_IMAGE);
    cv::imshow("Test Image", testImage);
    cv::waitKey(5000);
}