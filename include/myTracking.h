/**
 * @author Bozzon Alessandro.
 */
#ifndef TRACKING
#define TRACKING

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <FieldGeometryAndMask.h>
#include <Ball.h>


    /**
     * return a vector containing all the trackers initialized using the initial position of the balls
     * @param balls vector containing all the balls
     * @param first_frame cv::Mat containing the frame used for the initialization of the trackers
     */
    std::vector<cv::Ptr<cv::Tracker>> createTrackers(std::vector<Ball> balls, const cv::Mat first_frame);


    /**
     * update the values of the bounding box and the center of the circle of the ball based on the bounding boxes passed as argument
     * @param balls vector containing all the balls
     * @param rois vector containing the updated bounding boxes found using the tracker algorithm
     */
    void updateBallValues(std::vector<Ball> &balls, std::vector<cv::Rect> rois);

    /**
     * Return a vector containing 2 points for each ball: first point correspond to the position of the ball on the previous frame, second point correspond to the position of the ball on the current frame
     * @param balls vector containing all the balls
     * @param rois vector conatining all the bounding boxes obtained using the update of each single tracker
     */
    std::vector<cv::Point> computeBallMovement(std::vector<Ball> balls, std::vector<cv::Rect> rois);

#endif