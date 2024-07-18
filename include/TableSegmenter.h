/**
 * @author Alessandro Bozzon
 */
#ifndef TABLESEGMENTER_H
#define TABLESEGMENTER_H

#include <vector>

#include <opencv2/core.hpp>

class TableSegmenter{

    private:

    /**
     * Check if mask was already computed
     */
    bool maskComputed = false;

    /**
     * Check if corners were already computed
     */
    bool cornersComputed = false; 

    /**
     * Used to store already computed table mask
     */
    cv::Mat tableMask;

    /**
     * Used to store already computed table corners
     */
    std::vector<cv::Point2i> tableCorners;

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

    /**
     * Return a cv::Mat containing the draw of the 4 lines delimiting the playing field
     * @param field_contour cv::Mat containing the mask of the playing field
     */
    cv::Mat findFieldLines(const cv::Mat field_contour) const;

    /**
     * Return a vector containing the 4 points identifying the 4 corners of the playing field sorted as: top_left, top_right, bottom_right, bottom_left
     * @param approximate_field_lines cv::Mat containing the draw of the 4 lines delimiting the playing field from which the 4 corners will be computed
     */
    std::vector<cv::Point2i> findFieldCorners(const cv::Mat approximate_field_lines) const;

    public:

    /**
     * Returns the binary mask of the playing field
     * 
     * @param frame the current frame where mask is computed
     * 
     * @return the mask for the passed frame
     */
    cv::Mat getTableMask(const cv::Mat& frame);

    /**
     * Returns the binary maskcorners of the playing field
     * 
     * @param mask the mask of the playing field
     * 
     * @return a vector of integer points correspoinding to corners
     */
    std::vector<cv::Point2i> getFieldCorners(const cv::Mat& mask);

    /**
     * Returns masked frame
     * 
     * @param frame frame to be masked
     * @param mask mask to use
     * 
     * @return a frame with only masked values
     */
    cv::Mat getMaskedImage(const cv::Mat& frame, const cv::Mat& mask) const;

    
};

#endif