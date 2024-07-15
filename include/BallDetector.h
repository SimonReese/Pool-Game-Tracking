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
    std::vector<Ball> findBalls(const cv::Mat only_table_image, const cv::Mat field_contour, std::vector<cv::Point> boundaries_contours_poly, std::vector<cv::Point2i> sorted_corners);

    public:

    /**
     * Returns a vector of detected balls
     * 
     * @param image reference image used to detect balls
     * @param tableMask
     * @param tableContours
     * @param tableCorners
     */
    std::vector<Ball> detectBalls(const cv::Mat& image, const cv::Mat& tableMask, const std::vector<cv::Point>& tableContours, std::vector<cv::Point2i> tableCorners);


    cv::Mat drawBallsOnFieldMask(const cv::Mat field_mask, std::vector<Ball> balls);

};

#endif