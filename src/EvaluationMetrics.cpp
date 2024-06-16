#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include "EvaluationMetrics.h"

double EvaluationMetrics::computeIntersectionOverUnion(const std::vector<int>& firstBox, const std::vector<int>& secondBox) const{
    // Extract properties from vectors
    int x1, y1, w1 ,h1, class1;
    int x2, y2, w2, h2, class2;
    x1 = firstBox[0];
    y1 = firstBox[1];
    w1 = firstBox[2];
    h1 = firstBox[3];
    class1 = firstBox[4];

    x2 = secondBox[0];
    y2 = secondBox[1];
    w2 = secondBox[2];
    h2 = secondBox[3];
    class2 = secondBox[4];

    // Check if frames are intersecting
    if (x2 > x1 + w1 || y2 > y1 + h1){ // Box 2 is lower or too much to the right of box1
        // Box 2 is not overlapping box 1
        return 0;
    }
    else if (x2 + w2 < x1 || y2 + h2 < y1){ // Box 2 is upper or too much to the left of box1
        // Box 2 is not overlapping box 1
        return 0;
    }

    // Find coordinates of intersecting rectangle
    int Ax = std::max(x1, x2);
    int Ay = std::max(y1, y2);
    int Dx = std::min(x1 + w1, x2 + w2);
    int Dy = std::min(y1 + h1, y2 + h2);

    // Find size of intersection
    int width = Dx - Ax;
    int height = Dy - Ay;

    std::cout << "Sizes: " << width << "x" << height << std::endl;

    // Compute intersection area
    double intersection = width * height;
    std::cout << "Intersection: " << intersection << std::endl; 

    // Compute union area
    double un_ion = (w1*h1) + (w2*h2) - (intersection);
    std::cout << "Union: " << un_ion << std::endl;
    // Return IoU
    return intersection / un_ion;
}

std::vector<std::vector<int> > EvaluationMetrics::readBoundingBoxFile(std::string filePath) const{

    // Open file
    std::ifstream boundingBoxFile(filePath);

    // Check file is open
    if (!boundingBoxFile.is_open()){
        std::cerr << "Error. Could not open file " << filePath << std::endl;
    }

    // Vector of vectors representing the bounding boxes
    std::vector<std::vector<int> > boundingBoxes;
    // Properties of each bounding box (x, y) , width, height, class
    int x, y, w, h, cls;
    while(boundingBoxFile.peek() != EOF){
        std::vector<int> boundingBox;
        // Unpack line
        boundingBoxFile >> x >> y >> w >> h >> cls;
        // Discard newline
        boundingBoxFile.ignore();
        // Append elements in vector
        boundingBox.insert(boundingBox.end(), { x, y, w, h, cls });

        // Append box to vector of boxes
        boundingBoxes.push_back(boundingBox);
    }

    return boundingBoxes;
}

double EvaluationMetrics::evaluateBoundingBoxes(std::string trueFile, std::string predictedFile) const {

    // Create vectors to store files
    std::vector<std::vector<int>> trueBoudingBoxes = readBoundingBoxFile(trueFile);
    std::vector<std::vector<int>> predictedBoudingBoxes = readBoundingBoxFile(predictedFile);

    // Find how many boxes we predicted
    int boxes = predictedBoudingBoxes.size();
    std::cout << "Found " << boxes << " predictions" << std::endl;
    // Mean score
    double meanScore = 0;
    // Match each prediction against all groud truth and keep the best
    for(std::vector<int> boundingBox : predictedBoudingBoxes){
        double best_IoU = 0;
        for(std::vector<int> groundBox : trueBoudingBoxes){
            double IoU = computeIntersectionOverUnion(groundBox, boundingBox);
            std::cout<<"IoU was " << IoU;
            if (IoU > best_IoU) { best_IoU = IoU; std::cout << " keeped" << std::endl;}
            std::cout << std::endl;
        }
        meanScore += best_IoU;
    }
    // Compute mean score
    meanScore /= boxes;

    return meanScore;
}

double EvaluationMetrics::maskedIoU(const cv::Mat &maskedGroundTruth, const cv::Mat &maskedPrediction, int classes) const{

    // Construct vector to remap values
    std::vector<uchar> table(256);
    for (int i = 0; i < classes; i++){
        table[i] = pow(2, i);
    }

    // Remap values for ground truth and for predictions
    cv::Mat remappedGroundTruth, remappedPrediction;
    cv::LUT(maskedGroundTruth, table, remappedGroundTruth);
    cv::LUT(maskedPrediction, table, remappedPrediction);

    // Compute IoU for each class
    double meanIoU = 0;
    for (int i = 0; i < classes; i++){
        // Construct a matrix with all pixels of the same value of class value (remapped value)
        cv::Mat filter(maskedGroundTruth.rows, maskedGroundTruth.cols, CV_8UC1, cv::Scalar(table[i]));
        // Filter ground truth and prediction to have only one class
        cv::Mat classTruth, classPredictions;
        cv::bitwise_and(remappedGroundTruth, filter, classTruth);
        cv::bitwise_and(remappedPrediction, filter, classPredictions);
        // Compute intersection between ground truth and predictions
        cv::Mat intersection;
        cv::bitwise_and(classTruth, classPredictions, intersection);
        double intersectionArea = cv::countNonZero(intersection);
        // Compute union between classes
        cv::Mat u_nion;
        cv::bitwise_or(classTruth, classPredictions, u_nion);
        double u_nionArea = cv::countNonZero(u_nion);

        // Compute IoU
        meanIoU += (intersectionArea / u_nionArea );
    }
    meanIoU = meanIoU / classes;

    return meanIoU;
}

EvaluationMetrics::EvaluationMetrics(std::string groundTruthPath, std::string predictionPath)
{
    this->groundTruthPath = groundTruthPath;
    this->groundTruthPath = predictionPath;
}

double EvaluationMetrics::meanIoUMasked(std::string firstFile, std::string secondFile, int classes) const{

    cv::Mat groundTruth = cv::imread(firstFile, cv::IMREAD_GRAYSCALE);
    cv::Mat prediction = cv::imread(secondFile, cv::IMREAD_GRAYSCALE);

    return maskedIoU(groundTruth, prediction, classes);
}

double EvaluationMetrics::meanIoUMasked(const cv::Mat &firstImage, const cv::Mat &secondImage, int classes) const{
    return maskedIoU(firstImage, secondImage, classes);
}

double EvaluationMetrics::meanIoUtwoFiles(std::string firstFile, std::string secondFile)const {

    return evaluateBoundingBoxes(firstFile, secondFile);
}