#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "EvaluationMetrics.h"

int main(int argc, char* argv[]){
    if (argc != 1){
    if (argc < 3){
        std::cerr << "Error. Please provide paths to bounding box files. Missing " << 3 - argc << std::endl;
        return -1;
    }

    std::string firstPath = argv[1];
    std::string secondPath = argv[2];

    EvaluationMetrics metrics("", "");
    double meanIoU = metrics.meanIoUtwoFiles(firstPath, secondPath);
    std::cout << "Mean IoU between two:" << meanIoU << std::endl;
    }

    EvaluationMetrics metrics("", "");
    cv::Mat ground = cv::imread("../res/Dataset/game1_clip2/masks/frame_first.png", cv::IMREAD_GRAYSCALE);
    cv::Mat pred = cv::imread("../res/predictions/game1_clip2/masks/frame_first.png", cv::IMREAD_GRAYSCALE);
    std::vector<uchar> table(256);
    // For predictions
    table[0] = 0;
    table[255] = 1;
    table[127] = 2;
    // For truth
    table[1] = 2;
    table[2] = 2;
    table[3] = 2;
    table[4] = 2;
    table[5] = 1;
    cv::LUT(pred, table, pred);
    cv::LUT(ground, table, ground);
    double IoU = metrics.meanIoUMasked(ground, pred, 3);
    std::cout << "IoU masked:" << IoU << std::endl;
    return 0;
}