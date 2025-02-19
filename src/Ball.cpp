/**
 * @author Alessandro Bozzon
 */
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Ball.h"

Ball::Ball(){
    this->radius = 0;
    this->center = cv::Point(0, 0);
    this->type = Ball::BallType::UNKNOWN;
    this->bounding_box = cv::Rect(0,0,2,2);
    this->whiteRatio = -1.0;
}

Ball::Ball(cv::Vec3f circle_radius_and_center){
    this->radius = static_cast<int>(circle_radius_and_center[2]);
    this->center = cv::Point(static_cast<int>(circle_radius_and_center[0]), static_cast<int>(circle_radius_and_center[1]));
    this->type = Ball::BallType::UNKNOWN;
    this->bounding_box = cv::Rect(0,0,2,2);
    this->whiteRatio = -1.0;
}

Ball::Ball(cv::Vec3i circle_radius_and_center){
    this->radius = circle_radius_and_center[2];
    this->center = cv::Point(circle_radius_and_center[0], circle_radius_and_center[1]);
    this->type = Ball::BallType::UNKNOWN;
    this->bounding_box = cv::Rect(0,0,2,2);
    this->whiteRatio = -1.0;
}

Ball::Ball(int radius, cv::Point center){
    this->radius = radius;
    this->center = center;
    this->type = Ball::BallType::UNKNOWN;
    this->bounding_box = cv::Rect(0,0,2,2);
    this->whiteRatio = -1.0;
}

void Ball::setBoundingBox(cv::Rect bounding_box){
    this->bounding_box = bounding_box;
}

cv::Rect Ball::getBoundingBox() const{
    return bounding_box;
}

void Ball::setBallPosition(cv::Vec3i circle_radius_and_center){
    this->radius = circle_radius_and_center[2];
    this->center = cv::Point(circle_radius_and_center[0], circle_radius_and_center[1]);
}

void Ball::setBallPosition(int radius, cv::Point center){
    this->radius = radius;
    this->center = center;
}

cv::Vec3i Ball::getBallPosition(){
    return cv::Vec3i(this->center.x, this->center.y, this->radius);
}

cv::Point Ball::getBallCenter(){
    return center;
}


void Ball::setBallCenter(cv::Point center){
    this->center = center;
}


cv::Point Ball::getBallCenterInBoundingBox(){

    int center_x = this->center.x - bounding_box.x;
    int center_y = this->center.y - bounding_box.y;
    return cv::Point2i(center_x, center_y);

}

int Ball::getBallRadius(){
    return radius;
}

Ball::BallType Ball::getBallType() const{
    return type;
}

void Ball::setBallType(Ball::BallType type){
    this->type = type;
}

void Ball::setWhiteRatio(float whiteRatio){
    this->whiteRatio = whiteRatio;
}

float Ball::getWhiteRatio() const{
    return whiteRatio;
}

std::string Ball::typeToString(){
    std::string type;
    switch (this->type)
    {
    case Ball::BallType::FULL:
        return "FULL";
        break;
    case Ball::BallType::HALF:
        return "HALF";
        break;
    case Ball::BallType::WHITE:
        return "WHITE";
        break;
    case Ball::BallType::BLACK:
        return "BLACK";
        break;
    case Ball::BallType::UNKNOWN:
        return "UNKNOWN";
        break;
    default:
        return "UNKNOWN";
        break;
    }
}