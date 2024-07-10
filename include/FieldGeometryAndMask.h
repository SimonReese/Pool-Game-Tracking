/**
 * @author Bozzon Alessandro.
 */
#ifndef FIELDGEOMETRYANDMASK
#define FIELDGEOMETRYANDMASK
#include <opencv2/core/mat.hpp>
#include <string>
#include <Ball.h>

    /**
     * Return a vector containing the mean color for each of the 3 channels (mainly used with HSV color space) with the kernel centered on the center of the image
     * @param image image where to compute the mean of the color
     * @param kernel_size size of the square kernel used. height = width = kernel_size
     */    
    cv::Vec3b fieldMeanColor(const cv::Mat image, int kernel_size);

    /**
     * Return a cv::Mat containing the mask of the field without considering the balls
     * @param image image where to compute the mask from
     * @param mean_color vector containing the mean color to use for the selection of pixels that will form the mask
     */ 
    cv::Mat computeFieldMask(const cv::Mat image, cv::Vec3b mean_color);

    /**
     * Return a cv::Mat containing the draw of the 4 lines delimiting the playing field
     * @param field_contour cv::Mat containing the mask of the playing field
     */
    cv::Mat findFieldLines(const cv::Mat field_contour);

    /**
     * Return a vector containing the 4 points identifying the 4 corners of the playing field sorted as: top_left, top_right, bottom_right, bottom_left
     * @param approximate_field_lines cv::Mat containing the draw of the 4 lines delimiting the playing field from which the 4 corners will be computed
     */
    std::vector<cv::Point2i> findFieldCorners(const cv::Mat approximate_field_lines);

    /**
     * Return a vector of points that represent the polygon that bounds the playing field. Useful for the elimination of false positive performed by the BallsDetection class
     * @param sorted_corners vector containg the 4 sorted corners of the field
     * @param approximate_field_lines used only to obtain a new cv::Mat of the same size of the input image
     */
    std::vector<cv::Point> defineBoundingPolygon(std::vector<cv::Point2i> sorted_corners, const cv::Mat approximate_field_lines);

    /**
     * Return a cv::Mat containing the full mask of the field and the balls
     * @param field_mask cv::Mat containing only the field mask without the balls
     * @param balls vector containing all the detected balls, used for retrieving balls position and class in order to draw them on the mask
     * CURRENTLY DRAW BALLS ALL OF THE SAME CLASS --> NEED CLASSIFICATION TO COMPLETE!!!
     */
    cv::Mat drawBallsOnFieldMask(const cv::Mat field_mask, std::vector<Ball> balls);

#endif