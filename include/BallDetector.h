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

    std::vector<cv::Rect> findBoundingRectangles(const cv::Mat field_mask_and_balls);

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
     * @param tableMask
     * @param tableContours
     * @param tableCorners
     */
    std::vector<Ball> detectBalls(const cv::Mat& image, const cv::Mat& tableMask, std::vector<cv::Point2i> tableCorners);


    cv::Mat drawBallsOnFieldMask(const cv::Mat field_mask, std::vector<Ball> balls);


    std::vector<Ball> detectballsAlt(cv::Mat frame);

    std::vector<cv::Point> getTableContours() {return this->tableContours;};

    void saveMaskToFile(const cv::Mat &mask, std::vector<Ball> balls, std::string predictedMaskPath);

    void saveBoxesToFile(std::vector<Ball> balls, std::string predictedBBoxPath);
};

#endif