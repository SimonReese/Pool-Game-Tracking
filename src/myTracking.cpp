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
#include "myTracking.h"


std::vector<cv::Ptr<cv::Tracker>> createTrackers(std::vector<Ball> balls, const cv::Mat first_frame){

    std::vector<cv::Ptr<cv::Tracker>> trackers; /*vector that will contain all the single trakcers, one per ball*/
    
    for (int i = 0; i < balls.size(); i++){
        cv::Ptr<cv::Tracker> tr = cv::TrackerCSRT::create(); /*creation of the tracker*/
        tr->init(first_frame,balls[i].getBoundingBox()); /*initialization of the tracker using the bounding box of the ball*/
        trackers.push_back(tr);
    }

    return trackers;
}


void updateBallValues(std::vector<Ball> &balls, std::vector<cv::Rect> rois){
    for (int i = 0; i < rois.size(); i++){
        balls[i].setBoundingBox(rois[i]); //updates the coordinates of the bounding box

        cv::Vec3i temp = balls[i].getBallPosition();
        cv::Vec3i new_position = cv::Vec3i(static_cast<int>(rois[i].x+rois[i].width/2),static_cast<int>(rois[i].y+rois[i].height/2),temp[2]); /*computes the new coordinates of the center of the cirlce that defines the ball*/
        balls[i].setBallPosition(new_position); //updates the coordinates of the center of the circle that represent the ball
        
    }
    
}

std::vector<cv::Point> computeBallMovement(std::vector<Ball> balls, std::vector<cv::Rect> rois){
    std::vector<cv::Point> ballMovement;

    for (int i = 0; i < rois.size(); i++){
        ballMovement.push_back(cv::Point(balls[i].getBallPosition()[0],balls[i].getBallPosition()[1])); /*insert in the vector the position of the ball in the previous frame*/
        ballMovement.push_back(cv::Point(rois[i].x+rois[i].width/2,rois[i].y+rois[i].height/2)); /*insert in the vector the position of the ball in the current frame*/
    }

    return ballMovement;
    
}