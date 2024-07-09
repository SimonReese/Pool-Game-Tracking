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


std::vector<Ball> findBalls(const cv::Mat only_table_image, const cv::Mat field_contour, std::vector<cv::Point> boundaries_contours_poly, std::vector<cv::Point2i> sorted_corners);

void drawBallsHSVChannels(std::vector<Ball> balls, cv::Mat &image);

void drawBoundingBoxesHSVChannels(std::vector<Ball> balls, cv::Mat &image);

void writeBboxToFile(std::string bboxFileName,std::vector<Ball> balls); //when conmplete it will print both bounding box and class of every ball

std::vector<cv::Rect> findBoundingRectangles(const cv::Mat field_mask_and_balls);

#endif