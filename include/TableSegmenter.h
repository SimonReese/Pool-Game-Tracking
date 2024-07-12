/**
 * @author .
 * TODO: who will maintain this file?
 */

#ifndef TABLESEGMENTER_H
#define TABLESEGMENTER_H

#include <vector>

#include <opencv2/core.hpp>

class TableSegmenter{

    private:

    /**
     * Return a vector containing the mean color for each of the 3 channels (mainly used with HSV color space) with the kernel centered on the center of the image
     * @param image image where to compute the mean of the color
     * @param kernel_size size of the square kernel used. height = width = kernel_size
     */    
    cv::Vec3b fieldMeanColor(const cv::Mat& image, int kernel_size) const;

    /**
     * Return a cv::Mat containing the mask of the field without considering the balls
     * @param image image where to compute the mask from
     * @param mean_color vector containing the mean color to use for the selection of pixels that will form the mask
     */ 
    cv::Mat computeFieldMask(const cv::Mat image, cv::Vec3b mean_color) const;


    public:

    cv::Mat getTableMask(const cv::Mat& frame) const;

    /**
     * Returns the playing field corners
     * 
     * Returns a vector containing the 4 points identifying the 4 corners of the 
     * playing field sorted as: top_left, top_right, bottom_right, bottom_left
     * 
     */
    std::vector<cv::Point2i> findFieldCorners();

};

#endif