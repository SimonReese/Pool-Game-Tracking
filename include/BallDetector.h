/**
 * @author Alessandro Bozzon
 */
#ifndef BALLDETECTOR_H
#define BALLDETECTOR_H

#include <vector>

#include <opencv2/imgproc.hpp>

#include "Ball.h"

class BallDetector{

    private:

    /**
     * Return a vector containing the mean color for each of the 3 channels (mainly used with HSV color space) with the kernel centered on the center of the image
     * @param image image where to compute the mean of the color
     * @param kernel_size size of the square kernel used. height = width = kernel_size
     */    
    cv::Vec3b fieldMeanColor(const cv::Mat& image, int kernel_size)const;

    /**
     * Returns a vector of detected balls
     * 
     * @param only_table_image  hsv no blurred table image
     */
    std::vector<Ball> findBalls(const cv::Mat only_table_image, const cv::Mat field_contour, std::vector<cv::Point2i> sorted_corners);


    /**
     * Returns a vector of cv::Rect that contains all the bounding boxes found from the mask passed as argument
     * 
     * @param field_mask_and_balls  mask of the table with the balls
     */
    std::vector<cv::Rect> findBoundingRectangles(const cv::Mat field_mask_and_balls);


    /**
     * auxiliary function to compute the general bounding polygon starting from the 4 corners of the table
     * 
     * @param sorted_corners  4 corners pf the table
     * @param frame cv::Mat used inside the function to create a new cv::Mat of the same dimensions where lines are plotted in order to the find the bounding polygon from them
     */
    void defineBoundingPolygon(std::vector<cv::Point2i> sorted_corners, const cv::Mat frame);

    /**
     * Used to store table contours
     */
    std::vector<cv::Point> tableContours;

    public:

    /**
     * Returns a vector of detected balls
     * 
     * @param image reference image used to detect balls
     * @param tableMask mask of the table used to select only the part of the image that contains the field and put the rest of the pixels to value 0
     * @param tableCorners 4 corners of the table, useful for the removal of wrongly detected circles inside the private function findBalls
     */
    std::vector<Ball> detectBalls(const cv::Mat& image, const cv::Mat& tableMask, std::vector<cv::Point2i> tableCorners);

    /**
     * Returns a cv::Mat that contains the mask of the field with the balls.
     * Field color in the mask is not the final color in order to have an high contrast between balls and field.
     * This is done in order to make it easier for the function findBoundingRectangles to define the bounding boxes of the balls
     * 
     * @param field_mask mask of the field without the balls
     * @param balls vector of balls from where the info about the position of every detected ball on the field are retrieved
     */
    cv::Mat drawBallsOnFieldMask(const cv::Mat field_mask, std::vector<Ball> balls);

    /**
     * Returns a vector of detected balls, alternative function to detectBalls
     * 
     * @param frame cv::Mat that contains the frame where to detect the balls
     */
    std::vector<Ball> detectballsAlt(cv::Mat frame);


    std::vector<cv::Point> getTableContours() {return this->tableContours;};

    /**
     * Utility function to save the complete mask of the field on the disk
     * 
     * @param mask cv::Mat that contains the mask of the field without the balls
     * @param balls vector of balls that is used to retrieve position information about the balls in order to plot them on the mask
     * @param predictedMaskPath path where to save the mask that contains both the field and the balls
     */
    void saveMaskToFile(const cv::Mat &mask, std::vector<Ball> balls, std::string predictedMaskPath);

    /**
     * Utility function to save the bounding box and the class of each ball on a file for later evaluation metrics computation
     * 
     * @param balls vector of balls that is used to retrieve bounding box and class information in order to write them on file
     * @param predictedBBoxPath path where to save the info about the balls(bounding box and class)
     */
    void saveBoxesToFile(std::vector<Ball> balls, std::string predictedBBoxPath);
};

#endif