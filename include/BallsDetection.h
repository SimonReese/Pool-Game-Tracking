/**
 * @author Bozzon Alessandro.
 */
#ifndef BALLSDETECTION
#define BALLSDETECTION

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include "FieldGeometryAndMask.h"
#include "Ball.h"

    /**
     * Return a vector containing all the ball found on the field
     * @param only_table_image image where to look for the balls
     * @param field_contour mask of the field used to remove false positive balls inside and outside the playing field
     * @param boundaries_contours_poly polygon representing the contour of the playing field, used fo the removal of false positive balls inside the playing field
     * @param sorted_corners 4 corners of the playing field, used for the removal of false positive ball close to the corners of the field
     */
    std::vector<Ball> findBalls(const cv::Mat only_table_image, const cv::Mat field_contour, std::vector<cv::Point> boundaries_contours_poly, std::vector<cv::Point2i> sorted_corners);

    /**
     * Modify the image by drawing the circles corresponding to the balls
     * @param balls vector of balls from which circles position and radius are retrieved from
     * @param image input image where the circles will be drawn
     */
    void drawBallsHSVChannels(std::vector<Ball> balls, const cv::Mat &image);

    /**
     * Modify the image by drawing the boudning box around found balls
     * @param balls vector of balls from which bounding box parameters are retrieved from
     * @param image input image where the bounding boxes will be drawn
     */
    void drawBoundingBoxesHSVChannels(std::vector<Ball> balls, const cv::Mat &image);

    /**
     * Write bounding box and class of every ball found for future calculation of evaluation metrics
     * @param bboxFileName filename of the file where to write the parameters of the bounding boxes
     * @param balls vector of balls from which bounding box parameters and class value are retrieved from
     */
    void writeBboxToFile(std::string bboxFileName,std::vector<Ball> balls); //when conmplete it will print both bounding box and class of every ball

    /**
     * Return a vector containing all the bounding boxes found, one for each ball detected in the previous functions
     * @param field_mask_and_balls image containing the circles that identifies the balls
     */
    std::vector<cv::Rect> findBoundingRectangles(const cv::Mat field_mask_and_balls);

#endif