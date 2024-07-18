/**
 * @author Federico Adami
 */
#ifndef BALL_TRACKER_H
#define BALL_TRACKER_H

#include <opencv2/tracking.hpp>

#include "Ball.h"

/**
 * Class to track the different balls in the game
 */
class BallTracker {

private:

    /**
     * vector containing the balls in the game that are not moving
     */
    std::vector<Ball> stillBalls;


    /**
     * vector containing the balls in the game that are moving
     */
    std::vector<Ball> movingBalls;


    /**
     * vector containing a tracker for each moving ball
     */
    std::vector<cv::Ptr<cv::Tracker>> ballTrackers;


    /**
     * first frame for from the video of the game
    */
    cv::Mat firstFrame;

    /**
     * @brief creates a tracker for each moving ball
     * @param frame video frame used for the initialization of the trackers
     * @return the vector of trackers, which has one tracker for each moving ball
     */
    std::vector<cv::Ptr<cv::Tracker>> createTrackers(const cv::Mat &frame) const;

    /**
     * @brief update the values of the bounding box and the center of the ball based on the bounding boxes passed as argument
     * @param newBoundingBoxes vector of updated bounding boxes found using the tracking algorithm
     */
    void updateBallsCenterAndBoundingBox(const std::vector<cv::Rect> &newBoundingBoxes);

    /**
     * @brief used in the constructor to find the white ball which is the one expected to move
     * @param numTrackedBalls number of balls to consider when looking for the white ball
     */
    void findWhiteBall( int numTrackedBalls = 3);

    /**
     * @brief updates the movingBalls vector if a new moving ball is detected
     * @param frame the current video frame of the game
     */
    void updateMovingBalls(const cv::Mat& frame, float collosionDistance = 25.0);

    /**
     * @param first first ball to compare
     * @param second second ball to compare
     * @return true if first ball has greater whiteRatio than second ball
     */
    static bool compareWhiteRatio(Ball first, Ball second);

    /**
     * @param first first ball to compare
     * @param second second ball to compare
     * @return the distance of the centers between first and second
    */
    static float ballsDistance(Ball first, Ball second);

public:
    
    /**
     * @brief the only constructor for this class
     * @param firstFrame the first frame in the video 
     * @param gameBalls vector of the balls detected in the game
    */
    BallTracker(const cv::Mat firstFrame, const std::vector<Ball> gameBalls);

    /**
     * @brief updates the state moving balls using the provided frame;
     * @param currentFrame current frame of the video used to update the state of the balls
     * @param ballsToUpdate vector of balls whose status is to be updated
     * @return true if the update was successful and all the balls were tracked
    */
    bool update(const cv::Mat &currentFrame, std::vector<Ball> &ballsToUpdate);

};
#endif