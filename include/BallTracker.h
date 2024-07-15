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

std::vector<Ball> gameBalls;


std::vector<cv::Rect> rois;    
std::vector<cv::Point> ballMovement;
cv::VideoCapture cap;



// this is for future tracker improvement
//std::vector<Ball> trackedBalls;



public:
    

    //this solution is temporary of using the path
    BallTracker(std::string videoClipPath){
        if(videoClipPath.empty()){
            std::cerr << "Error: Empty video path." << std::endl;
            return;
        }

        this->cap = cv::VideoCapture(videoClipPath);
        // initialization of private variables
        ballTrackers = std::vector<cv::Ptr<cv::Tracker>>();
        gameBalls = std::vector<Ball>();
    }



    // !!!!!!!!!!this is just temporary until detection is implemented
    BallTracker(cv::VideoCapture &cap, std::vector<Ball> &balls){
        if(!cap.isOpened()){
            std::cerr << "Error: VideoCapture constructor failed" << std::endl;
            return;
        }
        if (balls.empty()){
            std::cerr << "Error: balls vector is empty" << std::endl;
            return;
        }

        //VideoCapture initialization
        this->cap = cap;

        // vector of balls in the game initialization
        this->gameBalls = balls;

        // trackers initialization
        cv::Mat first_frame;
        cap >> first_frame;
        ballTrackers = BallTracker::createTrackers(balls, first_frame);
    }

    /**
     * return a vector containing all the trackers initialized using the initial position of the balls
     * @param balls vector containing all the balls
     * @param first_frame cv::Mat containing the frame used for the initialization of the trackers
     */
    std::vector<cv::Ptr<cv::Tracker>> createTrackers(const std::vector<Ball> &balls, const cv::Mat &first_frame);


    /**
     * update the values of the bounding box and the center of the circle of the ball based on the bounding boxes passed as argument
     * @param balls vector containing all the balls
     * @param rois vector containing the updated bounding boxes found using the tracker algorithm
     */
    void updateBallValues(std::vector<Ball> &balls, const std::vector<cv::Rect> &rois);

    /**
     * Return a vector containing 2 points for each ball: first point correspond to the position of the ball on the previous frame, second point correspond to the position of the ball on the current frame
     * @param balls vector containing all the balls
     * @param rois vector conatining all the bounding boxes obtained using the update of each single tracker
     */
    std::vector<cv::Point> computeBallMovement(std::vector<Ball> balls, std::vector<cv::Rect> rois);


    /**
     * 
     * @brief trackes the moving balls in the provided frame;
     * 
    */

    void startTracking();


};
#endif