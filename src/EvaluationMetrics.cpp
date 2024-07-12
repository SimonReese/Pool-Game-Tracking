/**
 * @author Simone Peraro
 */
#include "EvaluationMetrics.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tuple> // Requires C++11
#include <algorithm>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "utils.h"



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
    boundingBoxFile.close();
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
        // Filter each mask with only class related values
        cv::Scalar filter(table[i]);
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
        double IoU = intersectionArea / u_nionArea;
        // DEBUG
        std::cout << "class " << i << ": " << IoU << "; ";
        meanIoU += IoU;
    }
    std::cout << std::endl;
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
        std::vector<std::string> frameNames = getFrameNames(gameFolder);

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
        std::vector<std::string> frameNames = getFrameNames(gameFolder); 

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

std::vector<std::string> EvaluationMetrics::getFrameNames(std::string gameFolder) const{
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
    return frameNames;
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

double EvaluationMetrics::meanIoUSegmentationREMAPPED(int classes) const{
    double globalIoU = 0;
    // For each game clip
    for(std::string gameFolder : this->gameFolders){
        // Get all frames
        std::vector<std::string> frameNames = getFrameNames(gameFolder);
        // For each frame
        std::cout << "Evaluating game " << gameFolder << ":" << std::endl;
        for(std::string frameName : frameNames){
            // Masks
            std::string groundTruthMask = this->datasetPath + "/" + gameFolder + "/" + this->masksFolder + "/" + frameName + ".png";
            std::string predictedMask = this->predictionsPath + "/" + gameFolder + "/" + this->masksFolder + "/" + frameName + ".png";
            // Images
            cv::Mat truth = cv::imread(groundTruthMask, cv::IMREAD_GRAYSCALE);
            cv::Mat predicted = cv::imread(predictedMask, cv::IMREAD_GRAYSCALE);

            // Remapping
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
            cv::LUT(truth, table, truth);
            cv::LUT(predicted, table, predicted);

            // Compute IoU
            std::cout << "\t";
            double mIoU = meanIoUMasked(truth, predicted, 3);
            std::cout << "->frame " << frameName << " mIoU: " << mIoU << "\n" << std::endl;
            globalIoU += mIoU;
        }
    }
    globalIoU /= this->gameFolders.size() * 2; // *2 since each game will have 2 frames evaluated
    std::cout << "Global mean IoU: " << globalIoU << std::endl;
    return globalIoU;
}

double EvaluationMetrics::meanIoUSegmentation(int classes) const{
    double globalIoU = 0;
    // For each game clip
    for(std::string gameFolder : this->gameFolders){
        // Get all frames
        std::vector<std::string> frameNames = getFrameNames(gameFolder);
        // For each frame
        std::cout << "Evaluating game " << gameFolder << ":" << std::endl;
        for(std::string frameName : frameNames){
            // Masks
            std::string groundTruthMask = this->datasetPath + "/" + gameFolder + "/" + this->masksFolder + "/" + frameName + ".png";
            std::string predictedMask = this->predictionsPath + "/" + gameFolder + "/" + this->masksFolder + "/" + frameName + ".png";
            // Compute IoU
            std::cout << "\t";
            double mIoU = meanIoUMasked(groundTruthMask, predictedMask, classes);
            std::cout << "->frame " << frameName << " IoU: " << mIoU << "\n" <<std::endl;
            globalIoU += mIoU;
        }
    }
    globalIoU /= this->gameFolders.size() * 2; // *2 since each game will have 2 frames evaluated
    std::cout << "Global mean IoU: " << globalIoU << std::endl;
    return globalIoU;
}

double EvaluationMetrics::meanIoUtwoFiles(std::string firstFile, std::string secondFile)const {

    return evaluateBoundingBoxes(firstFile, secondFile);
}

/**
 * Computes mean Average Precision (mAP) across ball classes
 * 
 * @param predictedFilePath path of predicted bounding boxes file
 * @param groundTruthPath path of ground truth bounding boxes file
 * @param classes total number of classes in predicted and ground truth files
 * 
 * @return the class wise mean average precision
 */
double EvaluationMetrics::computeMeanAveragePrecision(std::string predictedFilePath, std::string groundTruthPath, int classes) const{

    // 1. Open files and get vectors of bounding boxes
    // Create vectors to store files
    std::vector<std::vector<int>> trueBoudingBoxes = readBoundingBoxFile(predictedFilePath);
    std::vector<std::vector<int>> predictedBoudingBoxes = readBoundingBoxFile(groundTruthPath);

    // We want to store IoU for every bbox and store if prediction is correct
    // Each element of the vector will be a tuple of (IoU-score, predicted-class, is-correct-prediction)
    std::vector<std::tuple<double, int, bool> > scoresIoU;


    // 2. Compute best IoU for every predicted bb
    // We also need to count the number of ground truth for each class
    int countGT[classes+1]; // Skip index 0
    // Match each prediction against all groud truth and keep the best score
    for(std::vector<int> boundingBox : predictedBoudingBoxes){
        // Create new tuple to store values
        std::tuple<double, int, bool> current = std::make_tuple(0.0, boundingBox[4], false);

        for(std::vector<int> groundBox : trueBoudingBoxes){
            double IoU = computeIntersectionOverUnion(groundBox, boundingBox);
            if (IoU > std::get<0>(current)) { 
                std::get<0>(current) = IoU; // save IoU score
                std::get<2>(current) = groundBox[4] == boundingBox[4]; // save correct prediction
            }
            // Count ground truth
            countGT[groundBox[4]]++;
        }
        // Append tuple to vector of scores
        scoresIoU.push_back(current);
    }

    // 3. Order tuples by IoU score
    std::sort(scoresIoU.begin(), scoresIoU.end(), sortTupleKeysDescending);

    /**
     * Thershold for IoU
     * TODO: set parameter globally
     * TODO: split huge codeblock
     */
    double IoUThreshold = 0.5;

    double meanAveragePrecision = 0;
    // 4. Now, for each ball class, compute average precision
    for(int i = 1; i <= classes; i++){ // Ball class id goes in range 1-4
        
        double averagePrecision = 0; // Average precision

        double cumulativeTP = 0; // True positive
        double cumulativeFP = 0; // False positive
        double cumulativePrecision; // False positive
        double cumulativeRecall; // False positive
        std::map<double, double> curve; // values for Precision recall curve
        // Take all IoU scores
        for(std::tuple<double, int, bool> score : scoresIoU){
            // Discard other classes
            if (std::get<1>(score) != i){
                continue;
            }

            // Update TP and FP
            if (!std::get<2>(score) || std::get<0>(score) < IoUThreshold ){ // If class is incorrect or class is correct but the score is too low, we have a false positive 
                cumulativeFP++;
            } else { // otherwise, if class is good and score is good we have a true positive
                cumulativeTP++;
            }
            
            // Compute precision and recall
            cumulativePrecision = cumulativeTP / cumulativeTP + cumulativeFP;
            cumulativeRecall = cumulativeTP / countGT[i];
            
            // Save precision and recall in a map
            // Chek if a point was already present
            std::map<double, double>::iterator previousPrecision = curve.find(cumulativeRecall);
            if (previousPrecision == curve.end()){ // If no previous element found
                curve[cumulativeRecall] = cumulativePrecision; // Insert new element
            } else if (previousPrecision->second < cumulativePrecision) { // Otherwise if previous element found and prevoius value is lower
                previousPrecision->second = cumulativePrecision; // Update value in map
            }
            
        }

        // Pascal Voc 11 point interpolation
        std::map<double, double> interpolated; // map of interpolated values
        double bestSoFar = 0;
        double point = 1.0;
        std::map<double, double>::reverse_iterator rit = curve.rbegin();
        // Check map is not empty
        if (rit == curve.rend()){
            // Skip to next class
            continue;
        }
        double recall = rit->first;
        double precision = rit->second;
        while(point >= 0.0){
            // Did we pass a recall value?
            if (point <= recall){
                //Update best so far if needed
                if (bestSoFar < precision){
                    bestSoFar = precision;
                }
                // Move on to the next recall value
                rit++;
                // Update values
                recall = rit->first;
                precision = rit->second;
            }
            // Set current value
            interpolated[point] = bestSoFar;
            // Move to next point
            point -= 0.1;
        }

        // Now we have our 11 points to compute AP
        for(std::map<double, double>::iterator it = interpolated.begin(); it != interpolated.end(); it++){
            averagePrecision += it->second;
        }
        averagePrecision /= 11;
        meanAveragePrecision += averagePrecision;
    }
    meanAveragePrecision /= classes;

    return meanAveragePrecision;
}