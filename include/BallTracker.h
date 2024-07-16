/**
 * @author Federico Adami
 */
#ifndef BALL_TRACKER_H
#define BALL_TRACKER_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>


#include "FieldGeometryAndMask.h"
#include "Ball.h"

class BallTracker {

private:

    // vector with a tracker for each ball in the field
    std::vector<cv::Ptr<cv::Tracker>> ballTrackers;

    // vector containing all the balls in the field
    std::vector<Ball> gameBalls;

    std::vector<Ball> trackedBalls;

    cv::Mat firstFrame;

    /**
     * return a vector containing all the trackers initialized using the initial position of the balls
     * @param balls vector containing all the balls
     * @param first_frame cv::Mat containing the frame used for the initialization of the trackers
     */
    std::vector<cv::Ptr<cv::Tracker>> createTrackers(const cv::Mat &first_frame);

    /**
     * update the values of the bounding box and the center of the circle of the ball based on the bounding boxes passed as argument
     * @param balls vector containing all the balls
     * @param rois vector containing the updated bounding boxes found using the tracker algorithm
     */
    void updateBallsCenterAndBoundingBox(const std::vector<cv::Rect> &rois);


    /**
     * @brief computes the vector of the balls most probabily are going to move
     * 
     * 
    */
    void computeTrackedBalls();

    static bool compareBall(Ball a, Ball b);

    static float ballsDistance(Ball first, Ball second);

    void updateTracked(const cv::Mat& frame);

public:
    
    // BallTracker constructor
    BallTracker(const cv::Mat first_frame, const std::vector<Ball> balls);
    /**
     * @brief trackes the moving balls in the provided frame;
    */
    bool update(const cv::Mat &frame, std::vector<Ball> &ballsToUpdate);

};
#endif