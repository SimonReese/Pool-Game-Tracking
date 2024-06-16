/**
 * @author Simone Peraro.
 */
#ifndef EVALUATIONMETRICS
#define EVALUATIONMETRICS
#include <opencv2/core/mat.hpp>
#include <string>

class EvaluationMetrics{

private:

    // Path to training dataset containing ground truth mask and bounding boxes
    std::string groundTruthPath;

    // Path to predicted mask containing predicted mask and bounding boxes
    std::string predictionsPath;

    double computeIntersectionOverUnion(const std::vector<int>& groundTruth, const std::vector<int>& prediction) const;

    // Read a file of bounding boxes and return a vector of vectors eache representing a bounding box
    std::vector<std::vector<int> > readBoundingBoxFile(std::string filePath) const;

    // Compute mean IoU between two files
    double evaluateBoundingBoxes(std::string trueFile, std::string predictedFile) const;

    // Compute intersection over union between two masked images. Images must be single channel, uchar datatype.
    // Each image pixel has a single value corresponding to a class. Up to 8 classes are supported.
    double maskedIoU(const cv::Mat& maskedGroundTruth, const cv::Mat& maskedPrediction, int classes) const;


public: 
    
    /**
     * Constructor to initialize Evaluation class.
     * 
     * @param groundTruthPath path to the ground truth dataset folder
     * @param predictionPath path to the predictions folder. Ground truth and predictions foders must have same structre. More specifically, they must respect the folder structure provided as-is in the Dataset folder.
     * 
     */
    EvaluationMetrics(std::string groundTruthPath, std::string predictionPath);

    // TEMPORARY FUNCTION TO COMPUTE MEAN IoU BETWEEN TWO FILES
    double meanIoUtwoFiles(std::string firstFile, std::string secondFile)const ;

    // TEMPORARY FUNCTION TO COMPUTE MASKED IMAGES IOU
    double meanIoUMasked(std::string firstFile, std::string secondFile)const;

};

#endif