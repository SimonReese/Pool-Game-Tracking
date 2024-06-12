#include <vector>
#include "EvaluationMetrics.h"

EvaluationMetrics::EvaluationMetrics(std::string groundTruthPath, std::string predictionPath){
    this->groundTruthPath = groundTruthPath;
    this->groundTruthPath = predictionPath;
}

double EvaluationMetrics::computeIntersectionOverUnion(const std::vector<int>& groundTruth, const std::vector<int>& prediction){
    // Extract properties from vectors
    int x1, y1, w1 ,h1, class1;
    int x2, y2, w2, h2, class2;
    x1 = groundTruth[0];
    y1 = groundTruth[1];
    w1 = groundTruth[2];
    h1 = groundTruth[3];
    class1 = groundTruth[4];

    x2 = prediction[0];
    y2 = prediction[1];
    w2 = prediction[2];
    h2 = prediction[3];
    class2 = prediction[4];

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
