/**
 * @author Simone Peraro.
 */
#ifndef EVALUATIONMETRICS
#define EVALUATIONMETRICS

#include <string>
#include <tuple>

#include <opencv2/core/mat.hpp>


class EvaluationMetrics{

private:

    //************************************* CLASS ATTRIBUTES ******************************************** */

    /**
     * Dataset path.
     *
     * Path to the root of the dataset folder, containing source video frames and ground truth files.
     * Video frames names will be used as key names to find all other files.
     */
    std::string gameClipFolder;

    /**
     * Output folder path.
     *
     * Path to the root of the predictions folder, containing predictions files.
     */
    std::string outputFolder;

    /**
     * Subfolder containing bounding boxes (both in gameClipFolder and outputFolder)
     */
    const std::string boundingBoxesFolder;

    /**
     * Subfolder containing masks images (both in gameClipFolder and outputFolder)
     */
    const std::string masksFolder;

    /**
     * Subfolder containing frames (only in gameClipFolder folder)
     */
    const std::string framesFolder;

    /**
     * Vector of all named frames found in frames subdirectoryspo
     */
    std::vector<std::string> framenames;

    /**
     * Extension for the bbox files
     */
    const std::string frameFileExtension = ".png";

    /**
     * Extension for the masks files
     */
    const std::string maskFileExtension = ".png";

    /**
     * Extension for the bbox files
     */
    const std::string bboxFileExtension = "_bbox.txt";

    /**
     * Output metrics filename
     */
    const std::string metricsFileName = "metrics.txt";

    //************************************* BOUNDING BOXES COMPUTATIONS ************************************** */


    /**
     * Read a file containing a list of bounding boxes.
     * 
     * @param filePath path to file containing bounding boxes
     * 
     * @return returns a vector of vector elements composed by four integers: (x, y, width, height, class)
     */
    std::vector<std::vector<int> > readBoundingBoxFile(std::string filePath) const;

    /**
     * Computes IoU between bounding boxes
     * 
     * In order to compute mAP, we need IoU values for the bounding boxes. To compute IoU between two 
     * bounding boxes, we first check if the two are actually overlapping, and then we compute the 
     * intersected area finding the maximum x, y values among the top left corners of the boxes, and the minimum w, h sizes.
     * Union area will be the sum of the two rectangle areas minus the intersection.
     * 
     * @param firstBox the first bounding box
     * @param secondBox the second bounding box
     * 
     * @return IoU of the two bounding boxes
     */
    double boxesIoU(const std::vector<int>& firstBox, const std::vector<int>& secondBox) const;

    /**
     * Compute IoU between two mask images.
     * 
     * This function performs IoU between two mask images, where each pixel in the mask has a value V
     * corresponding to the class of the pixel (background = 0, playing field = 5, white ball = 1, etc...).
     * In order to compute IoU, we remap each value V to the result of pow(2, v) (that is 0 -> 1, 
     * 1 -> 2, 2 -> 4, 3 -> 8, etc...). In binary format, those value will not have overlapping 1's (1, 10, 100, etc..)
     * and in this way we can easily filter out all the pixels belonging to a different class from the one considered 
     * (that is, we perform bitwise_and operation between the filter value V for the class considered and the mask).
     * Finally, we can easily compute: 
     * - Intersection, as bitwise_and operations between the two masks
     * - Union, as bitwise_or between the two masks
     * Area is obtained by counting non zero pixels after each operations.
     * Since masks images store single byte values, up to 7 classes are supported.
     * 
     * @param trueMask ground truth mask image
     * @param predictedMask predicted mask image
     * @param classes number of classes where IoU is computed (must be < 8)
     * 
     * @return a vector with IoU for each class
     * 
     * @throw `std::logic_error` if more than 7 classes are requested.
     */
    std::vector<double> masksIoU(const cv::Mat& trueMask, const cv::Mat& predictedMask, int classes) const;

    /**
     * Evaluate integrity of game clip folder.
     * 
     * We must be sure that the game clip folder exists, that the frames, masks and bounding boxes subfolders exits,
     * and that for each frame in frames subfolder, a corresponding bounding box and mask exits in subfolders.
     * 
     * @throw `std::invalid_argument` if any of the required folders or files are missing. 
     */
    void checkGameClipFolderIntegrity();

    /**
     * Evaluate integrity of game output folder.
     * 
     * We check that the output folder exists and has subfolders, otherwise we create missing ones
     * 
     * @throw `std::runtime_error` if a folder is missing but we couldn't create one. 
     */
    void checkOutputFolderIntegrity();

public:

    /**
     * @return a vector of all path-to frame files in game clip folder
     */
    std::vector<std::string> getFrameFiles();

    /**
     * @return a vector of all path-to true mask files in game clip folder
     */
    std::vector<std::string> getTrueMaskFiles();

    /**
     * @return a vector of all path-to true bounding boxes files in game clip folder
     */
    std::vector<std::string> getTrueBoundingBoxFiles();

    /**
     * @return a vector of all path-to predicted mask files in output folder
     */
    std::vector<std::string> getPredictedMaskFiles();
    
    /**
     * @return a vector of all path-to predicted bounding boxes files in output folder
     */
    std::vector<std::string> getPredictedBoundingBoxFiles();


    /**
     * Constructor to initialize Evaluation class.
     * 
     * This constructor will take path to game clip folder and path to output folder as parameters, along with names for masks, frames and bounding boxes subfolders.
     * @param gameClipFolderPath path to the game clip folder
     * @param outputFolderPath path to the output folder.
     * @param framesFolder (optional) name of the subfolder containing frames images. Default value is `frames`.
     * @param maksFolder (optional) name of the subfolder containing masks images. Default value is `masks`.
     * @param boundingBoxesFolder (optional) name of the subfolder containing bounding boxes .txt files. Default value is `bounding_boxes`.
     * 
     * @throw `std::invalid_argument` if dataset folder or predictions folder are not accessible
     * @throw `std::logic_error` if dataset folder or predictions folder are not consistent with folders structure
     */
    EvaluationMetrics(std::string gameClipFolderPath, 
                    std::string outputFolderPath, 
                    std::string framesFolder = "frames", 
                    std::string masksFolder = "masks", 
                    std::string boundingBoxesFolder = "bounding_boxes"
                    );

    /**
     * Computes segmentation IoU for each frame, in each game folder, for each class.
     * 
     * @param trueMask path to true mask file
     * @param predictedMask path to predicted mask file
     * @param classes the number of classes to consider (default: 6)
     */
    double computeMasksIoU(std::string trueMask, std::string predictedMask, int classes = 6) const;

    /**
     * Computes mean Average Precision (mAP) across ball classes
     * 
     * @param predictedFilePath path of predicted bounding boxes file
     * @param groundTruthPath path of ground truth bounding boxes file
     * @param classes total number of classes in predicted and ground truth bounding boxes files (default: 4)
     * 
     * @return the class wise mean average precision
     */
    double computeMeanAveragePrecision(std::string predictedFilePath, std::string groundTruthPath, int classes = 4) const;
};

#endif