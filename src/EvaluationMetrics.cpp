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

#include "FilesystemUtils.h"


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


double EvaluationMetrics::boxesIoU(const std::vector<int>& firstBox, const std::vector<int>& secondBox) const{
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
    double un_ion = (w1*h1) + (w2*h2) - (intersection);
    // Return IoU
    return intersection / un_ion;
}


std::vector<double> EvaluationMetrics::masksIoU(const cv::Mat &trueMask, const cv::Mat &predictedMask, int classes) const{
    // Make sure we are not overflowing a byte
    if(classes > 7){
        throw std::logic_error("Error. Cannot perform IoU in this way for more than 7 classes.");
    }

    std::vector<double> scoresIoU;

    // Construct vector to remap values
    std::vector<uchar> table(256);
    for (int i = 0; i < classes; i++){
        table[i] = pow(2, i);
    }

    // Remap values for ground truth and for predictions
    cv::Mat remappedGroundTruth, remappedPrediction;
    cv::LUT(trueMask, table, remappedGroundTruth);
    cv::LUT(predictedMask, table, remappedPrediction);

    // Compute IoU for each class
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

        // Compute IoU and append to vector
        double IoU = intersectionArea / u_nionArea;
        scoresIoU.push_back(IoU);
    }

    return scoresIoU;
}


/**
 * Evaluate integrity of game clip folder.
 * 
 * We must be sure that the game clip folder exists, that the frames, masks and bounding boxes subfolders exits,
 * and that for each frame in frames subfolder, a corresponding bounding box and mask exits in subfolders.
 * 
 * @throw `std::invalid_argument` if any of the required folders or files are missing. 
 */
void EvaluationMetrics::checkGameClipFolderIntegrity(){
    // We need to check if the game clip folder is reachable
    if (!cv::utils::fs::exists(this->gameClipFolder)){
        throw std::invalid_argument("Error. Game clip folder " + this->gameClipFolder + " not found.");
    }
    // We need to check also the subfolders integrity
    std::string framesSubfolder = cv::utils::fs::join(this->gameClipFolder, this->framesFolder);
    std::string masksSubfolder = cv::utils::fs::join(this->gameClipFolder, this->masksFolder);
    std::string bboxesSubfolder = cv::utils::fs::join(this->gameClipFolder, this->boundingBoxesFolder);
    if (!cv::utils::fs::exists(framesSubfolder)){
        throw std::invalid_argument("Error. Missing " + framesSubfolder + " subfolder.");
    }
    if (!cv::utils::fs::exists(masksSubfolder)){
        throw std::invalid_argument("Error. Missing " + masksSubfolder + " subfolder.");
    }
    if (!cv::utils::fs::exists(bboxesSubfolder)){
        throw std::invalid_argument("Error. Missing " + bboxesSubfolder + " subfolder.");
    }

    // We need to check files exists in each subfolder
    std::vector<std::string> framenames;
    cv::utils::fs::glob_relative(framesSubfolder, "", framenames);
    // Store frames filenames (without extensions) in a class vector
    for(std::string framename : framenames){
        this->framenames.push_back(framename.erase(framename.find_last_of('.')));
    }

    // Now we check that all files exists in the corresponding directories
    for (std::string framename : this->framenames){
        // Construct file names
        std::string bboxFileName = framename + this->bboxFileExtension;
        std::string maskFileName = framename + this->maskFileExtension;
        // Construct file paths
        std::string bboxFilePath = cv::utils::fs::join(bboxesSubfolder, bboxFileName);
        std::string maskFilePath = cv::utils::fs::join(masksSubfolder, maskFileName);


        if(!cv::utils::fs::exists(bboxFilePath)){
            throw std::invalid_argument("Error. Missing file " + bboxFilePath + ".");
        }
        if(!cv::utils::fs::exists(maskFilePath)){
            throw std::invalid_argument("Error. Missing file " + maskFilePath + ".");
        }
    }
    // At this point we checked that gameClip folder exists, correspondent gamefolders exists and that 
    // for eac frame in frames subfolder, a corresponding mask and bounding box file exists
}

/**
 * Evaluate integrity of game output folder.
 * 
 * We check that the output folder exists and has subfolders, otherwise we create missing ones
 * 
 * @throw `std::runtime_error` if a folder is missing but we cant create one. 
 */
void EvaluationMetrics::checkOutputFolderIntegrity(){
    // Construct subfolder paths
    std::string masksSubfolder = cv::utils::fs::join(this->outputFolder, this->masksFolder);
    std::string bboxesSubfolder = cv::utils::fs::join(this->outputFolder, this->boundingBoxesFolder);
    bool created;
    // We check if folder exists
    if(!cv::utils::fs::exists(this->outputFolder)){
        // We can create a corresponding folder containing all directories
        created = cv::utils::fs::createDirectories(masksSubfolder);
        if (!created) {throw std::runtime_error("Error. " + masksSubfolder + " is missing and couldn't create one.");} 
        created = cv::utils::fs::createDirectories(bboxesSubfolder);
        if (!created) {throw std::runtime_error("Error. " + bboxesSubfolder + " is missing and couldn't create one.");}
        return;
    }

    // If folder exists, we check subfolders or we create missing ones
    if(!cv::utils::fs::exists(masksSubfolder)){
        created = cv::utils::fs::createDirectory(masksSubfolder);
        if (!created) {throw std::runtime_error("Error. " + masksSubfolder + " is missing and couldn't create one.");} 
    }
    if(!cv::utils::fs::exists(bboxesSubfolder)){
        created = cv::utils::fs::createDirectory(bboxesSubfolder);
        if (!created) {throw std::runtime_error("Error. " + bboxesSubfolder + " is missing and couldn't create one.");} 
    }
}


// --------------------- P U B L I C  F U N C T I O N S ---------------------------------------------


std::vector<std::string> EvaluationMetrics::getFrameFiles(){
    std::vector<std::string> fullNames;
    for (std::string framename : this->framenames){
        std::string fullName = cv::utils::fs::join(this->gameClipFolder, this->framesFolder) +
                                framename + this->frameFileExtension;
        fullNames.push_back(fullName);
    }
    return fullNames;
}

std::vector<std::string> EvaluationMetrics::getTrueMaskFiles(){
    std::vector<std::string> fullNames;
    for (std::string framename : this->framenames){
        std::string fullName = cv::utils::fs::join(this->gameClipFolder, this->masksFolder) +
                                framename + this->maskFileExtension;
        fullNames.push_back(fullName);
    }
    return fullNames;
}

std::vector<std::string> EvaluationMetrics::getTrueBoundingBoxFiles(){
    std::vector<std::string> fullNames;
    for (std::string framename : this->framenames){
        std::string fullName = cv::utils::fs::join(this->gameClipFolder, this->boundingBoxesFolder) +
                                framename + this->bboxFileExtension;
        fullNames.push_back(fullName);
    }
    return fullNames;
}

std::vector<std::string> EvaluationMetrics::getPredictedMaskFiles(){
    std::vector<std::string> fullNames;
    for (std::string framename : this->framenames){
        std::string fullName = cv::utils::fs::join(this->outputFolder, this->masksFolder) +
                                framename + this->maskFileExtension;
        fullNames.push_back(fullName);
    }
    return fullNames;
}

std::vector<std::string> EvaluationMetrics::getPredictedBoundingBoxFiles(){
    std::vector<std::string> fullNames;
    for (std::string framename : this->framenames){
        std::string fullName = cv::utils::fs::join(this->outputFolder, this->boundingBoxesFolder) +
                                framename + this->bboxFileExtension;
        fullNames.push_back(fullName);
    }
    return fullNames;
}

EvaluationMetrics::EvaluationMetrics(std::string gameClipFolder, std::string outputFolder, std::string framesFolder, std::string masksFolder, std::string boundingBoxesFolder)
    : gameClipFolder{gameClipFolder}, outputFolder{outputFolder}, framesFolder{framesFolder}, masksFolder{masksFolder}, boundingBoxesFolder{boundingBoxesFolder} {
    
    // Check if game clip folder is consistent
    checkGameClipFolderIntegrity();

    // Check if output folder is consistent
    checkOutputFolderIntegrity();
}


double EvaluationMetrics::computeMasksIoU(std::string trueMask, std::string predictedMask, int classes) const {
    // Open the two maks
    cv::Mat truth = cv::imread(trueMask, cv::IMREAD_GRAYSCALE);
    cv::Mat predicted = cv::imread(predictedMask, cv::IMREAD_GRAYSCALE);
    // Compute IoU
    std::vector<double> scoresIoU = masksIoU(truth, predicted, classes);
    double globalIoU = 0;
    for(int i = 0; i < classes; i++){
        std::cout << "IoU mask score for class " << i << " is: " << scoresIoU[i] << std::endl;
        globalIoU += scoresIoU[i];
    }
    globalIoU /= classes;
    std::cout << "Mean IoU is: " << globalIoU << std::endl;
    return globalIoU;
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

    // We also need to count the number of ground truth for each class
    int countGT[classes+1]; // Skip index 0
    for (int i = 0; i < classes+1; i++){
        // Count ground truth
        countGT[i] = 0;
    }
    for(std::vector<int> groundBox : trueBoudingBoxes){
        // Count ground truth
        countGT[groundBox[4]]++;
    }

    // 2. Compute best IoU for every predicted bb
    // Match each prediction against all groud truth and keep the best score
    for(std::vector<int> boundingBox : predictedBoudingBoxes){
        // Create new tuple to store values
        std::tuple<double, int, bool> current = std::make_tuple(0.0, boundingBox[4], false);

        for(std::vector<int> groundBox : trueBoudingBoxes){
            double IoU = boxesIoU(groundBox, boundingBox);
            if (IoU > std::get<0>(current)) { 
                std::get<0>(current) = IoU; // save IoU score
                std::get<2>(current) = groundBox[4] == boundingBox[4]; // save correct prediction
            }
        }
        // Append tuple to vector of scores
        scoresIoU.push_back(current);
    }
    // DEBUG
    std::cout << "Total ground truth elements for each class:";
    for (int i = 1; i <= classes; i++){
        std::cout << " " << countGT[i];
    }
    std::cout << std::endl;

    // 3. Order tuples by IoU score
    std::sort(scoresIoU.begin(), scoresIoU.end(), sortTupleKeysDescending);
    
    /**
     * Threshold for IoU
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
            //std::cout << "AP point is: " << previousPrecision->first << ", " << previousPrecision->second << std::endl;
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

                if (rit != curve.rend()){rit++;}
                // Update values if end not reached
                if (rit != curve.rend()){
                    recall = rit->first;
                    precision = rit->second;
                } 
            }
            // Set current value
            interpolated[point] = bestSoFar;
            //std::cout << "Set interpolation " << point << " at " << bestSoFar << std::endl;
            // Move to next point
            point -= 0.1;
            
        }
        // Now we have our 11 points to compute AP
        for(std::map<double, double>::iterator it = interpolated.begin(); it != interpolated.end(); it++){
            averagePrecision += it->second;
        }
        averagePrecision /= 11;
        std::cout << "Average precision for class " << i << " is: " << averagePrecision << std::endl;
        meanAveragePrecision += averagePrecision;
    }
    meanAveragePrecision /= classes;
    std::cout << "Mean average precision is: " << meanAveragePrecision << std::endl;
    return meanAveragePrecision;
}
