/**
 * @author Federico Adami
 */
#ifndef BALL_CLASSIFIER_H
#define BALL_CLASSIFIER_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>


#include <BallsDetection.h>
#include <FieldGeometryAndMask.h>
#include <Ball.h>
#include <FilesystemUtils.h>


const std::string CUTOUT_DIR = "cutouts";

/**
 * @brief classifies given balls
*/
class BallClassifier{
    private:
        // classification threshold is 9.9% of the total number of pixels in the image
        static constexpr float CLASSIFICATION_TH = 0.099;

        /*setting the ranges for every HLS on which perform the thresholding*/ 

        // HUE thresholds
        static const int H_LOW = 0;
        static const int H_HIGH = 180;

        // SATURATION thresholds
        static const int L_LOW = 150;
        static const int L_HIGH = 255;

        // VALUE thresholds
        static const int S_LOW = 30;
        static const int S_HIGH = 255;

    public:

        /**
        * Default constructor for initializing class.
        */
        BallClassifier();

        /**
         * Function to calculate the ratio of white pixels in the binary image.
         * @param inputBinaryImage binary image obtained by thresholding HLS image
         * @return ratio of white pixels in the binary image
        */
        static float calculateWhitePixelsRatio(const cv::Mat &inputBinaryImage);

        /**
         * Function to classify a ball as white, black, half, or full based on the ratio of white
         * @param cutOutImage image of a ball to be classified
         * @return the type of the ball and the ratio of white pixels in the binary image
        */

        static std::pair<Ball::BallType, float> preliminaryBallClassifier(cv::Mat &cutOutImage);

        /**
         * Function to identify and classify the black and white balls from a game
         * @param ballsSet the set of balls in the fullGameImage
         * @param fullGameImage the full size image of the game to be analyzed
        */
        static void secondaryBallClassifier(std::vector<Ball> &ballsSet, const cv::Mat &fullGameImage);

};




/**
 * Function to show the HLS channels and binary image of a given image
 * @param channelsHlsImage vector of HLS channels of the input image
 * @param binaryImage binary image obtained by thresholding HLS image
*/
void showHlsChannelsandBinary(const std::vector<cv::Mat> &channelsHlsImage, const cv::Mat &binaryImage, std::string windowName);

/**
 * Function to evaluate the classification accuracy of the ball classifier on a class of balls in a game, in considers just full and half balls
 * @param datasetFolder name of the dataset folder
 * @param gameFolder name of the folder for the specific game we want to evaluate
 * @param ballClassFolder subfolder of game folder containing balls of the same class
 * @return number of balls wrongly classified
*/
int evaluateBallsSet(const std::string datasetFolder, const std::string gameFolder, const std::string ballClassFolder);

/**
 * Function to evaluate the classification accuracy of the ball classifier on all games in a given dataset
 * @param datasetFolder name of the dataset folder
*/
void evaluteGames(std::string datasetFolder);

/**
 * Function to save an image in a given folder
*/
void saveTofile(const cv::Mat &inputImage, std::string imageName, std::string outputFolder);


/**
 * Function to extract just a ball image cutout from a given image
*/
void cutOutBalls(const cv::Mat &inputImage, const std::vector<Ball> &balls, const std::string &gameFolder);


#endif 