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

    // Compute intersection area
    double intersection = width * height;

    // Compute union area
    double un_ion = (w1*h1) + (w2*h2);
    
    // Return IoU
    return intersection / un_ion;
}

std::vector<std::vector<int>> EvaluationMetrics::readBoundingBoxFile(std::string filePath) const{

    // Open file
    std::ifstream boundingBoxFile(filePath);

    // Check file is open
    if (!boundingBoxFile.is_open()){
        std::cerr << "Error. Could not open file " << filePath << std::endl;
    }

    // Vector of vectors representing the bounding boxes
    std::vector<std::vector<int>> boundingBoxes;
    // Properties of each bounding box (x, y) , width, height, class
    int x, y, w, h, cls;
    while(boundingBoxFile.peek() != EOF){
        std::vector<int> boundingBox;
        // Unpack line
        boundingBoxFile >> x >> y >> w >> h >> cls;
        // Append elements in vector
        boundingBox.insert(boundingBox.end(), {x, y, w, h, cls});

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
    // Mean score
    double meanScore = 0;
    // Match each prediction against all groud truth and keep the best
    for(std::vector<int> boundingBox : predictedBoudingBoxes){
        double best_IoU = 0;
        for(std::vector<int> groundBox : trueBoudingBoxes){
            double IoU = computeIntersectionOverUnion(groundBox, boundingBox);
            if (IoU > best_IoU) { best_IoU = IoU;}
        }
        meanScore += best_IoU;
    }
    // Compute mean score
    meanScore /= boxes;

}

EvaluationMetrics::EvaluationMetrics(std::string groundTruthPath, std::string predictionPath)
{
    this->groundTruthPath = groundTruthPath;
    this->groundTruthPath = predictionPath;
}

double EvaluationMetrics::meanIoUtwoFiles(std::string firstFile, std::string secondFile){

    
}




