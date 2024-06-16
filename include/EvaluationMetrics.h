/**
 * @author Simone Peraro.
 */
#ifndef EVALUATIONMETRICS
#define EVALUATIONMETRICS
#include <opencv2/core/mat.hpp>
#include <string>

class EvaluationMetrics{

private:

    /**
     * Dataset path.
     *
     * Path to the root of the dataset folder, containing source video frames and ground truth files.
     * Video frames names will be used as key names to find all other files.
     */
    std::string datasetPath;

    /**
     * Predictions path.
     *
     * Path to the root of the predictions folder, containing predictions files.
     */
    std::string predictionsPath;

    /**
     * Game folders
     * 
     * List of different games directories. In Dataset and Predictions paths it is expected 
     * to have one or more game folders, where each game folder will contain other folders and files needed to evaluate metrics.
     */
    std::vector<std::string> gameFolders;

    /**
     * Name for all folders containing bounding boxes (both for dataset and predictions)
     */
    const std::string boundingBoxesFolder;

    /**
     * Name for all folders containing masks images (both for dataset and predictions)
     */
    const std::string masksFolder;

    /**
     * Name for all folders containing frames (only in dataset folder)
     */
    const std::string framesFolder;

    double computeIntersectionOverUnion(const std::vector<int>& groundTruth, const std::vector<int>& prediction) const;

    // Read a file of bounding boxes and return a vector of vectors eache representing a bounding box
    std::vector<std::vector<int> > readBoundingBoxFile(std::string filePath) const;

    // Compute mean IoU between two files
    double evaluateBoundingBoxes(std::string trueFile, std::string predictedFile) const;

    // Compute intersection over union between two masked images. Images must be single channel, uchar datatype.
    // Each image pixel has a single value corresponding to a class. Up to 8 classes are supported.
    double maskedIoU(const cv::Mat& maskedGroundTruth, const cv::Mat& maskedPrediction, int classes) const;

    void checkDatasetFolder();

    void checkPredictionsFolder();
public: 

    /**
     * Constructor to initialize Evaluation class.
     * 
     * This constructor will take paths to dataset and predictions folders as parameters, along with names for masks, frames and bounding boxes folders.
     * @param groundTruthPath path to the ground truth dataset folder
     * @param predictionPath path to the predictions folder. Ground truth and predictions foders must have same structre. More specifically, they must respect the folder structure provided as-is in the Dataset folder.
     * @param framesFolder name of folder containing frames images. Default value is `frames`.
     * @param maksFolder name of folder containing masks images. Default value is `masks`.
     * @param boundingBoxesFolder name of folder containing bounding boxes .txt files. Default value is `bounding_boxes`.
     */
    EvaluationMetrics(std::string datasetPath, std::string predictionsPath, std::string framesFolder = "frames", std::string masksFolder = "masks", std::string boundingBoxesFolder = "bounding_boxes");

    // TEMPORARY FUNCTION TO COMPUTE MEAN IoU BETWEEN TWO FILES
    double meanIoUtwoFiles(std::string firstFile, std::string secondFile)const ;

    // TEMPORARY FUNCTION TO COMPUTE MASKED IMAGES IOU
    double meanIoUMasked(std::string firstFile, std::string secondFile, int classes)const;

    // TEMPORARY FUNCTION TO COMPUTE MASKED IMAGES IOU
    double meanIoUMasked(const cv::Mat& firstImage, const cv::Mat& secondImage, int classes)const;

};

#endif