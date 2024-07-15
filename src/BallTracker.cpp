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
#include "BallTracker.h"


std::vector<cv::Ptr<cv::Tracker>> BallTracker::createTrackers(const std::vector<Ball> &balls, const cv::Mat &first_frame){
    
    std::vector<cv::Ptr<cv::Tracker>> trackers; /*vector that will contain all the single trakcers, one per ball*/
    
    for (int i = 0; i < balls.size(); i++){
        cv::Ptr<cv::Tracker> tr = cv::TrackerCSRT::create(); /*creation of the tracker*/
        tr->init(first_frame, balls[i].getBoundingBox()); /*initialization of the tracker using the bounding box of the ball*/
        trackers.push_back(tr);
    }

    return trackers;
}


void BallTracker::updateBallValues(std::vector<Ball> &balls, const std::vector<cv::Rect> &rois){
    for (int i = 0; i < rois.size(); i++){
        balls[i].setBoundingBox(rois[i]); //updates the coordinates of the bounding box

        cv::Vec3f temp = balls[i].getBallPosition();
        cv::Vec3f new_position = cv::Vec3f(static_cast<float>(rois[i].x+rois[i].width/2),static_cast<float>(rois[i].y+rois[i].height/2),temp[2]); /*computes the new coordinates of the center of the cirlce that defines the ball*/
        balls[i].setBallPosition(new_position); //updates the coordinates of the center of the circle that represent the ball
    }
    
}

std::vector<cv::Point> BallTracker::computeBallMovement(std::vector<Ball> balls, std::vector<cv::Rect> rois){
    
    std::vector<cv::Point> ballMovement;
    for (int i = 0; i < rois.size(); i++){
        ballMovement.push_back(balls[i].getBallCenter());/*insert in the vector the position of the ball in the previous frame*/

        ballMovement.push_back(cv::Point(rois[i].x+rois[i].width/2,rois[i].y+rois[i].height/2)); /*insert in the vector the position of the ball in the current frame*/
    }

    return ballMovement;
    
}

void BallTracker::startTracking(){

    std::vector<cv::Rect> rois;
    std::vector<cv::Point> ballMovement;
    cv::Mat frame;

    for ( ;; ){

        
        // get frame from the video
        this->cap >> frame;
        
        // stop the program if no more images
        if(frame.rows==0 || frame.cols==0)
        break;
        
        rois.clear();
        ballMovement.clear();

        //update the tracking result
        for (int h = 0; h < this->ballTrackers.size(); h++){
            cv::Rect roi;
            this->ballTrackers[h]->update(frame,roi);
            rois.push_back(roi);
        }

        ballMovement = BallTracker::computeBallMovement(this->gameBalls, rois); /*balls still have the position relative to the previous frame*/ /*this vector contains pair of point that represent the ball movement from a frame to the next*/

        BallTracker::updateBallValues(this->gameBalls,rois);

        //debug test to see if bounding box and center of circle that defines ball gets updated by the function
        for (int g = 0; g < this->gameBalls.size(); g++){
            cv::rectangle(frame, this->gameBalls[g].getBoundingBox(), cv::Scalar(255,255,255),1,cv::LINE_AA);
            //cv::circle(frame,cv::Point(static_cast<int>(balls[g].getBallPosition()[0]),static_cast<int>((balls[g].getBallPosition()[1]))),static_cast<int>(balls[g].getBallPosition()[2]), cv::Scalar(255,255,255),-1);
        }

        // show image with the tracked object
        cv::imshow("tracker",frame);

        //quit on ESC button
        if(cv::waitKey(1)==27)break;

        
        //output.write(frame);
    }

}