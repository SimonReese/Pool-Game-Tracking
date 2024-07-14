#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "EvaluationMetrics.h"

int main(int argc, char* argv[]){
    if (argc != 1){
    if (argc < 3){
        std::cerr << "Error. Please provide paths to dataset and predictions folders. " << std::endl;
        return -1;
    }

    std::string datasetPath = argv[1];
    std::string predictionsPath = argv[2];

    //EvaluationMetrics metrics(datasetPath, predictionsPath);
    EvaluationMetrics metrics("../res/Dataset/", "../res/predictions/");
    //metrics.meanIoUSegmentationREMAPPED(3);
    
    double mAP = metrics.computeMeanAveragePrecision(predictionsPath, datasetPath);
    std::cout << "mAP:" << mAP << std::endl;
    return 0;
    }
    return -1;
    EvaluationMetrics metrics("", "");
    cv::Mat ground = cv::imread("../res/Dataset/game1_clip2/masks/frame_first.png", cv::IMREAD_GRAYSCALE);
    cv::Mat pred = cv::imread("../res/predictions/game1_clip2/masks/frame_first.png", cv::IMREAD_GRAYSCALE);
    
    double IoU = metrics.meanIoUMasked(ground, pred, 3);
    std::cout << "IoU masked:" << IoU << std::endl;
    return 0;
}