/**
 * @author Bozzon Alessandro.
 */
#ifndef BALL
#define BALL

#include <opencv2/core.hpp>


/**
 * Class to represent a ball during the game.
 */
class Ball{

private:
    // center and radius of the circle that identifies the ball
    cv::Vec3f circle_radius_and_center;

    // Class of the ball
    int ball_class;

    // measures of the bounding box that encloses the ball
    cv::Rect bounding_box;

public:

    Ball(cv::Vec3f circle_radius_and_center);

    void setBallClass(int ball_class);

    int getBallClass();

    void setBoundingBox(cv::Rect bounding_box);

    cv::Rect getBoundingBox();

    void setBallPosition(cv::Vec3f circle_radius_and_center);

    cv::Vec3f getBallPosition();

};
#endif