#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "FieldGeometryAndMask.h"
#include "Ball.h"
#include "BallTracker.h"


BallTracker::BallTracker(const cv::Mat first_frame, const std::vector<Ball> balls){
        if (balls.empty()){
            std::cerr << "Error: balls vector is empty" << std::endl;
            return;
        }

        if(first_frame.empty()){
            std::cerr << "Error: first video frame is empty" << std::endl;
            return;
        }
    
        // vector of balls in the game initialization
        this->gameBalls = balls;

        // trackers initialization
        this->ballTrackers = BallTracker::createTrackers(first_frame);
}

std::vector<cv::Ptr<cv::Tracker>> BallTracker::createTrackers(const cv::Mat &first_frame){

    if(first_frame.empty()) throw std::runtime_error("Error: request initialization on empty frame");
    
    std::vector<cv::Ptr<cv::Tracker>> trackers; /*vector that will contain all the single trakcers, one per ball*/
    for (int i = 0; i < this->gameBalls.size(); i++){

        cv::Ptr<cv::Tracker> tr = cv::TrackerCSRT::create(); /*creation of the tracker*/
        tr->init(first_frame, this->gameBalls[i].getBoundingBox()); /*initialization of the tracker using the bounding box of the ball*/
        
        trackers.push_back(tr);
    }
    
    return trackers;
}


void BallTracker::updateBallsCenterAndBoundingBox(const std::vector<cv::Rect> &rois){
    
    for (int i = 0; i < rois.size(); i++){
        //updates the coordinates of the center of the circle that represent the ball
        this->gameBalls[i].setBallCenter(cv::Point(rois[i].x+rois[i].width/2, rois[i].y+rois[i].height/2)); 
        this->gameBalls[i].setBoundingBox(rois[i]);
    }
    
}


std::vector<Ball> BallTracker::update(const cv::Mat &frame){
    if(frame.empty()) throw std::runtime_error("Error: requeste update on empty frame");
    std::vector<cv::Rect> rois;
    
    //update the tracking result
    for (int h = 0; h < this->ballTrackers.size(); h++){
        cv::Rect roi;
        this->ballTrackers[h]->update(frame,roi);
        rois.push_back(roi);
    }

    

    BallTracker::updateBallsCenterAndBoundingBox(rois);

    return this->gameBalls;
}