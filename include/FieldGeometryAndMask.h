/**
 * @author Bozzon Alessandro.
 */
#ifndef FIELDGEOMETRYANDMASK
#define FIELDGEOMETRYANDMASK
#include <opencv2/core/mat.hpp>
#include <string>
#include <Ball.h>

    
cv::Vec3b fieldMeanColor(const cv::Mat image, int kernel_size);

cv::Mat computeFieldMask(const cv::Mat image, cv::Vec3b mean_color);

cv::Mat findFieldLines(const cv::Mat field_contour);

std::vector<cv::Point2i> findFieldCorners(const cv::Mat approximate_field_lines);

std::vector<cv::Point> defineBoundingPolygon(std::vector<cv::Point2i> sorted_corners, const cv::Mat approximate_field_lines);

//currently all balls are of the same class!!!
cv::Mat drawBallsOnFieldMask(const cv::Mat field_mask, std::vector<Ball> balls);

#endif