#include <iostream>
#include <string>
#include <vector>
#include "EvaluationMetrics.h"

int main(int argc, char* argv[]){

    if (argc < 3){
        std::cerr << "Error. Please provide paths to bounding box files. Missing " << 3 - argc << std::endl;
        return -1;
    }

    std::string firstPath = argv[1];
    std::string secondPath = argv[2];

    EvaluationMetrics metrics("", "");
    double meanIoU = metrics.meanIoUtwoFiles(firstPath, secondPath);
    std::cout << "Mean IoU between two:" << meanIoU << std::endl;

    return 0;
}