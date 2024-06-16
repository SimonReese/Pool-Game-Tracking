#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

void EvaluationMetrics::checkDatasetFolder(){

    // Each game folder should have a frames folder, a masks folder and a bounding boxes folder
    for (std::string gameFolder : this->gameFolders){
        std::string gameFolderPath = this->datasetPath + "/" + gameFolder + "/";
        // Frames
        if (!cv::utils::fs::isDirectory(gameFolderPath + this->framesFolder)){
            throw std::logic_error("Error. Dataset folder " + this->datasetPath + " is missing the frames folder " + gameFolderPath + this->framesFolder + ". Please provide this dataset frame folder.");
        }
        // Masks
        if (!cv::utils::fs::isDirectory(gameFolderPath + this->masksFolder)){
            throw std::logic_error("Error. Dataset folder " + this->datasetPath + " is missing the masks folder " + gameFolderPath + this->masksFolder + ". Please provide this dataset masks folder.");
        }
        // Bounding boxes
        if (!cv::utils::fs::isDirectory(gameFolderPath + this->boundingBoxesFolder)){
            throw std::logic_error("Error. Dataset folder " + this->datasetPath + " is missing the bounding boxes folder " + gameFolderPath + this->boundingBoxesFolder + ". Please provide this dataset masks folder.");
        }

        // For each frame in frames folder, it should be a corresponding file in masks and bounding boxes folder
        std::vector<std::string> frameNames;
        cv::utils::fs::glob_relative(gameFolderPath + this->framesFolder, "", frameNames);
        // Remove extension
        for (std::vector<std::string>::iterator it = frameNames.begin(); it != frameNames.end(); it++){
            // Find last .extension occurrence
            int index = it->rfind(".");
            // Remove .extension from filename
            *it = it->erase(index, std::string::npos);
        }
        // Check in masks and bounding_boxes folders
        for(std::string frameName : frameNames){
            std::string maskName = gameFolderPath + this->masksFolder + "/" + frameName + ".png";
            std::string bboxName = gameFolderPath + this->boundingBoxesFolder + "/" + frameName + "_bbox" + ".txt";
            if(!cv::utils::fs::exists(maskName)){
                throw std::logic_error("Error. File " + maskName + " doesn't exists.");
            }
            if(!cv::utils::fs::exists(bboxName)){
                throw std::logic_error("Error. File " + bboxName + " doesn't exists.");
            }
        }
        // At this point, the game folder will have all required subfolders and files
    } // END FOR
}

void EvaluationMetrics::checkPredictionsFolder(){

    // Check integrity of predictions folder
    for (std::string gameFolder: this->gameFolders){
        std::string gameFolderPath = this->predictionsPath + "/" + gameFolder;

        // Game folder must exists
        if (!cv::utils::fs::isDirectory(gameFolderPath)){
            throw std::logic_error("Error. Predictions folder " + this->datasetPath + " is missing the game folder " + gameFolderPath + ".");
        }

        // Get frame name in frames folder FROM DATASET FOLDER
        std::vector<std::string> frameNames;
        cv::utils::fs::glob_relative(this->datasetPath + "/" + gameFolder + "/" + this->framesFolder, "", frameNames);
        // Remove extension
        for (std::vector<std::string>::iterator it = frameNames.begin(); it != frameNames.end(); it++){
            // Find last .extension occurrence
            int index = it->rfind(".");
            // Remove .extension from filename
            *it = it->erase(index, std::string::npos);
        }

        // For each frame, mask file and bounding box files must exists
        for(std::string frameName: frameNames){
            std::string maskName = gameFolderPath + "/" + this->masksFolder + "/" + frameName + ".png";
            std::string bboxName = gameFolderPath + "/" + this->boundingBoxesFolder + "/" + frameName + "_bbox" + ".txt";
            if(!cv::utils::fs::exists(maskName)){
                throw std::logic_error("Error. File " + maskName + " doesn't exists.");
            }
            if(!cv::utils::fs::exists(bboxName)){
                throw std::logic_error("Error. File " + bboxName + " doesn't exists.");
            }
        }
        // At this point, predictions folder should be consistent with dataset folder
    } // END FOR

}

EvaluationMetrics::EvaluationMetrics(std::string datasetPath, std::string predictionsPath, std::string framesFolder, std::string masksFolder, std::string boundingBoxesFolder)
    : datasetPath{datasetPath}, predictionsPath{predictionsPath}, framesFolder{framesFolder}, masksFolder{masksFolder}, boundingBoxesFolder{boundingBoxesFolder} {
    // Check whether folders are reachable
    if(!cv::utils::fs::isDirectory(this->datasetPath)){
        throw std::invalid_argument("Error. Dataset folder " + this->datasetPath + " not found.");
    }

    if(!cv::utils::fs::isDirectory(this->predictionsPath)){
        throw std::invalid_argument("Error. Predictions folder " + this->predictionsPath + " not found.");
    }

    // List game folders
    cv::utils::fs::glob_relative(this->datasetPath, "", this->gameFolders, false, true);
    // Remove elements which are not directories
    for(std::vector<std::string>::iterator it = this->gameFolders.begin(); it != gameFolders.end(); ){
        if (!cv::utils::fs::isDirectory(this->datasetPath + "/" + *it))
            it = this->gameFolders.erase(it);
        else
            it++;
    }
    
    // Check at least one game folder is present
    if(this->gameFolders.size() < 1){
        throw std::logic_error("Error. Dataset folder " + this->datasetPath + " has not games subfolders. At least one game folder is required");
    }

    // Check contents of dataset folder
    checkDatasetFolder();

    // Check consistency of predictions folder
    checkPredictionsFolder();
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