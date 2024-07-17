#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "Ball.h"
#include "BallTracker.h"


BallTracker::BallTracker(const cv::Mat firstFrame, const std::vector<Ball> balls){
        if (balls.empty()) throw std::runtime_error("Error: balls vector is empty");

        if(firstFrame.empty()) throw std::runtime_error("Error: first video frame is empty");

        this->firstFrame = firstFrame;
        // vector of balls in the game initialization
        this->gameBalls = balls;

        //tracked vector initialization
        computeTrackedBalls();

        for(Ball ball : this->trackedBalls){
            std::cout << "Ball " << ball.typeToString() << std::endl;
        }
        
        // trackers initialization
        this->ballTrackers = BallTracker::createTrackers(firstFrame);
}

std::vector<cv::Ptr<cv::Tracker>> BallTracker::createTrackers(const cv::Mat &firstFrame){

    if(firstFrame.empty()) throw std::runtime_error("Error: request initialization on empty frame");
    
    std::vector<cv::Ptr<cv::Tracker>> trackers; /*vector that will contain all the single trakcers, one per ball*/
    for (int i = 0; i < this->trackedBalls.size(); i++){

        cv::Ptr<cv::Tracker> tr = cv::TrackerCSRT::create(); /*creation of the tracker*/
        tr->init(firstFrame, this->trackedBalls[i].getBoundingBox()); /*initialization of the tracker using the bounding box of the ball*/
        
        trackers.push_back(tr);
    }
    
    return trackers;
}


void BallTracker::updateBallsCenterAndBoundingBox(const std::vector<cv::Rect> &rois){
    
    for (int i = 0; i < rois.size(); i++){
        //updates the coordinates of the center of the circle that represent the ball
        this->trackedBalls[i].setBallCenter(cv::Point(rois[i].x+rois[i].width/2, rois[i].y+rois[i].height/2)); 
        this->trackedBalls[i].setBoundingBox(rois[i]);
    }
    
}

bool BallTracker::compareBall(Ball a, Ball b){ 
    return (a.getWhiteRatio() > b.getWhiteRatio()); 
} 

void BallTracker::computeTrackedBalls(){
    const int numTrackedBalls = 2;
    std::sort(this->gameBalls.begin(),this->gameBalls.end(), compareBall);

    std::copy(this->gameBalls.begin(), this->gameBalls.begin() + numTrackedBalls, std::back_inserter(this->trackedBalls));  

    // for(Ball ball : this->gameBalls){
    //     std::cout << "check: " << ball.getWhiteRatio() << ball.typeToString() << std::endl;
    // }
    this->gameBalls.erase(this->gameBalls.begin(),this->gameBalls.begin() + numTrackedBalls);
}


bool BallTracker::update(const cv::Mat &frame, std::vector<Ball> &ballsToUpdate){
    if(frame.empty()) throw std::runtime_error("Error: requeste update on empty frame");
    std::vector<cv::Rect> rois;
    
    
    //update the tracking result
    for (int h = 0; h < this->ballTrackers.size(); h++){
        cv::Rect roi;
        
        bool ballFound = this->ballTrackers[h]->update(frame,roi);
        if(!ballFound){
            std::cout << "saltato" << std::endl;
            return false;
        }

        rois.push_back(roi);
    }

    BallTracker::updateBallsCenterAndBoundingBox(rois);
    BallTracker::updateTracked(frame);

    std::vector<Ball> allBalls;

    std::cout << "trackedBalls: " << trackedBalls.size() << std::endl;

    allBalls.insert( allBalls.end(), trackedBalls.begin(), trackedBalls.end() );
    allBalls.insert( allBalls.end(), gameBalls.begin(), gameBalls.end() );  

    ballsToUpdate.clear();
    std::copy(allBalls.begin(), allBalls.end(), std::back_inserter(ballsToUpdate));

    return true;
}

float BallTracker::ballsDistance(Ball first, Ball second){
        return sqrt(pow(first.getBallCenter().x - second.getBallCenter().x, 2) + pow(first.getBallCenter().y - second.getBallCenter().y, 2));
}

void BallTracker::updateTracked(const cv::Mat& frame){
    // update the tracked balls based on their distance with the game balls
    for(Ball moving : this->trackedBalls){
        for(int i = 0; i < this->gameBalls.size(); i++){

            if(BallTracker::ballsDistance(moving, this->gameBalls[i]) < 25.0){
                this->trackedBalls.push_back(this->gameBalls[i]);
                this->gameBalls.erase(this->gameBalls.begin() + i);
                cv::Ptr<cv::Tracker> tr = cv::TrackerCSRT::create(); /*creation of the tracker*/
                tr->init(this->firstFrame, this->trackedBalls.back().getBoundingBox()); /*initialization of the tracker using the bounding box of the ball*/
                this->ballTrackers.push_back(tr);
            }

        }
    }

}

