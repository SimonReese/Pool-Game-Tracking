#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Ball.h"

Ball::Ball(cv::Vec3f circle_radius_and_center){
    this->circle_radius_and_center = circle_radius_and_center;
}

void Ball::setBallClass(int ball_class){
    this->ball_class = ball_class;
}

int Ball::getBallClass(){
    return ball_class;
}

void Ball::setBoundingBox(cv::Rect bounding_box){
    this->bounding_box = bounding_box;
}

cv::Rect Ball::getBoundingBox(){
    return bounding_box;
}

void Ball::setBallPosition(cv::Vec3f circle_radius_and_center){
    this->circle_radius_and_center = circle_radius_and_center;
}

cv::Vec3f Ball::getBallPosition(){
    return circle_radius_and_center;
}
