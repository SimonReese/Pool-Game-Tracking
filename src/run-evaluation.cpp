/**
 * @author Simone Peraro
 * 
 * This runner will evaluate metrics of the system.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "EvaluationMetrics.h"
#include "TableSegmenter.h"
#include "Ball.h"
#include "BallDetector.h"
#include "BallClassifier.h"

/**
 * This main runner requires a path to the gameX_clipX folder (with masks, frames and bboxes folders) and
 * a path to a user defined folder where outputs will be stored 
 * (the output folder and relatives subfolders will be created if not existent, using the name provided by user via the second command line argument)
 */
int main(int argc, char* argv[]){

    if (argc < 3){
        std::cerr << "Error. Please provide path to gameX_clipX folder and path to output folder" << std::endl;
        return -1;
    }

    // Save gameclip path and output path
    std::string clipFolder = argv[1];
    std::string outputFolder = argv[2];

    // Instantiate evaluation metrics class
    EvaluationMetrics evaluate(clipFolder, outputFolder);
    std::vector<std::string> frames = evaluate.getFrameFiles();
    std::vector<std::string> masks = evaluate.getTrueMaskFiles();
    std::vector<std::string> bboxes = evaluate.getTrueBoundingBoxFiles();

    std::vector<std::string> omasks = evaluate.getPredictedMaskFiles();
    std::vector<std::string> obboxes = evaluate.getPredictedBoundingBoxFiles();

    // Create name for output frame with bboxes
    std::string boxesFrameName = cv::utils::fs::join(outputFolder, "bboxes-");

    // Compute outputs algorithm
    for (int i = 0; i < frames.size(); i++){
        std::string framePath = frames[i];
        // Prepare output file paths
        std::string predictedMaskPath = omasks[i];
        std::string predictedBBoxPath = obboxes[i];

        cv::Mat frame = cv::imread(framePath);

        // 1. Get table mask
        TableSegmenter segmenter;
        cv::Mat mask = segmenter.getTableMask(frame);

        // 2. Get table corners
        std::vector<cv::Point2i> corners = segmenter.getFieldCorners(mask);

        // 3. Detect balls
        BallDetector ballDetector;
        std::vector<Ball> balls = ballDetector.detectBalls(frame, mask, corners);
        //std::vector<Ball> balls = ballDetector.detectballsAlt(frame);
        
        // 4. Classify balls
        BallClassifier ballClassifier;
        balls = ballClassifier.classify(balls, frame);

        // 5. Draw classified balls over mask image
        ballDetector.saveMaskToFile(mask, balls, predictedMaskPath); // Must merge balls class and table mask

        // 6. Save balls bounding boxes
        ballDetector.saveBoxesToFile(balls, predictedBBoxPath); // Must save bboxes to file

        // 7. Write bounding boxes to frame and save file
        std::string frameName = frames[i].substr(frames[i].find_last_of('/') + 1);
        std::cout << "Saving bounding boxes to " << boxesFrameName + frameName << std::endl;
        for(Ball ball : balls){
            cv::rectangle(frame, ball.getBoundingBox(), cv::Scalar(51, 255, 255));
        }
        cv::imwrite(boxesFrameName + frameName, frame);
    }
    // Save metrics to file
    std::fstream outFile(cv::utils::fs::join(outputFolder, "metrics.txt"), std::fstream::out);
    outFile << "true_frame_name mIoU mAP" << std::endl;
    // Perform metrics evaluation
    for (int i = 0; i < frames.size(); i++){
        std::string trueMaskPath = masks[i];
        std::string trueBBoxPath = bboxes[i];
        
        // Prepare output file paths
        std::string predictedMaskPath = omasks[i];
        std::string predictedBBoxPath = obboxes[i];

        // Output
        std::cout << "\nRunning over frame " << frames[i] << ":" << std::endl;

        double mIoU = evaluate.computeMasksIoU(trueMaskPath, predictedMaskPath);
        double mAP = evaluate.computeMeanAveragePrecision(trueBBoxPath, predictedBBoxPath);
        outFile << frames[i] << " " << mIoU << " " << mAP << std::endl;
    }

    return 0;

    /*

    

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
    */
}