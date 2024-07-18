/**
 * @author Simone Peraro.
 */

#include "Draw.h"

#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Ball.h"


cv::Mat Draw::drawOver(const cv::Mat &background, const cv::Mat &overlapping, const cv::Point position){
    // Copy input image 
    cv::Mat result = background.clone();
    // Compute top left corner position of the overlapping object
    cv::Point corner(
        position.x - overlapping.cols / 2,
        position.y - overlapping.rows / 2
    );
    // Create region of interest
    cv::Rect rect(corner, cv::Size(overlapping.cols, overlapping.rows));
    cv::Mat region = result(rect); // reference to a section of resulting image

    // Convert overlapping image to grayscale and create inverted mask
    cv::Mat mask, invMask;
    mask = cv::Mat(overlapping.size(), CV_8UC1, cv::Scalar(255));
    cv::bitwise_not(mask, invMask);
    // Use inverted mask to cut region, setting pixels to 0
    cv::Mat cutted;
    cv::bitwise_and(region, region, cutted, invMask);

    // Sum overlapping image to blacked areas in the cutted region
    cv::Mat sum;
    cv::add(overlapping, cutted, sum);

    // Copy summed image to region
    sum.copyTo(region);
    return result;
}


Draw::Draw(){
    // Noting to do yet
}


cv::Mat Draw::updateDrawing(std::vector<Ball> balls){
    
    // Check that the perspective correction matrix was already computed
    if(!this->computedPerspective){
        throw std::runtime_error("Error. Requested a drawing update, but the perspective correction matrix was never computed.");
    }
    
    // Correct perspective for balls points
    std::vector<cv::Point2f> centers;
    for(Ball ball : balls){
        centers.push_back(ball.getBallCenter());
    }
    cv::perspectiveTransform(centers, centers, this->perspectiveTrasformation);

    // Draw balls
    cv::Mat drawing; // Drawing result to be returned
    drawing = this->drawingNoBalls.clone();
    for (int i = 0; i < balls.size(); i++){
        Ball ball = balls[i];
        cv::Point center = centers[i];

        // Select color according to ball type
        cv::Scalar color;
        switch (ball.getBallType())
        {
        case Ball::BallType::WHITE:
            color = cv::Scalar(255, 255, 255);
            break;
        case Ball::BallType::BLACK:
            color = cv::Scalar(0, 0, 0);
            break;
        case Ball::BallType::FULL:
            color = cv::Scalar(255, 0, 0);
            break;
        case Ball::BallType::HALF:
            color = cv::Scalar(153, 153, 255);
            break;
        default:
            color = cv::Scalar(255, 51, 153);
            break;
        }
        // Draw trajectory points
        cv::circle(this->drawingNoBalls, center, 2, cv::Scalar(102, 255, 255), -1);
        // Draw pngs
        //drawing = drawOver(this->drawingNoBalls, ballPNG, center);
        cv::circle(drawing, center, 10, color, -1);
    }

    return drawing;
}


void Draw::computePrespective(const std::vector<cv::Point>& corners){
    
    std::vector<cv::Point2f> srcCoord;
    // Convert from point2i to point2f
    cv::Mat(corners).copyTo(srcCoord);

    // We want to check if table is oriented horizontaly or vertically
    float horiz = cv::norm(corners[0] - corners[1]);
    float vert = cv::norm(corners[1] - corners[2]);

    // We build destination points accordingly
    std::vector<cv::Point2f> destCoord;
    cv::Mat result;
    cv::Size dsize;
    // Out table backgroud will have size of 340x650
    if (horiz / vert < this->tableRatio){
        // We will build a vertical pool table 
        int w = this->tableDrawSize.width;
        int h = this->tableDrawSize.height;
        destCoord = {
            cv::Point(0, 0) + cv::Point(this->padding, this->padding),      // Top left corner, must add padding
            cv::Point(w -1, 0) + cv::Point( - this->padding, this->padding), // Top right corner, must subtract and add padding
            cv::Point(w -1, h -1) - cv::Point(this->padding, this->padding),  // Top right corner, must subtract padding
            cv::Point(0, h -1) + cv::Point(this->padding, - this->padding)   // Top right corner, must add and subtract padding
        };
        dsize = cv::Size(w, h);
        result = cv::Mat(dsize, CV_8UC3);
        this->drawingNoBalls = cv::imread(this->verticalTablePath);
        
    }
    else {
        // We build a horizontal pool table
        int w = this->tableDrawSize.height; // Swap dimensions
        int h = this->tableDrawSize.width;
        destCoord = {
            cv::Point(0, 0) + cv::Point(this->padding, this->padding),      // Top left corner, must add padding,
            cv::Point(w -1, 0) + cv::Point( - this->padding, this->padding), // Top right corner, must subtract and add padding
            cv::Point(w -1, h -1) - cv::Point(this->padding, this->padding),  // Top right corner, must subtract padding,
            cv::Point(0, h -1) + cv::Point(this->padding, - this->padding)   // Top right corner, must add and subtract padding
        };
        dsize = cv::Size(w, h);
        result = cv::Mat(dsize, CV_8UC3);
        this->drawingNoBalls = cv::imread(this->horizontalTablePath);
        
    }
    
    // Compute transformation matrix
    this->perspectiveTrasformation = cv::getPerspectiveTransform(srcCoord, destCoord);
    this->computedPerspective = true;
}

cv::Mat Draw::displayOverlay(const cv::Mat &frame, const cv::Mat &drawing){
    // We want to put drawing in bottom left corner of the frame
    // We also may want to resize the image
    cv::Mat scaledDrawing;
    cv::resize(drawing, scaledDrawing, cv::Size(), Draw::overlayResizeRatio, Draw::overlayResizeRatio);

    // Compute the top left corner position of drawing over frame image
    cv::Point corner = cv::Point(
        0,
        frame.rows - scaledDrawing.rows
    );

    // Compute position of center of drawing
    cv::Point center = cv::Point(
        scaledDrawing.cols / 2,
        scaledDrawing.rows / 2
    );

    // Compute position of the center with respect to frame
    cv::Point position = center + corner;

    // Draw overlay
    cv::Mat result = Draw::drawOver(frame, scaledDrawing, position);

    return result;
}
