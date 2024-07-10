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

    /**
     * Default constructor for initializing class.
     * 
     * @param circle_radius_and_center center point and radius of the circle representing the ball
     */
    Ball(cv::Vec3f circle_radius_and_center);

    /**
     * Set the class of the ball
     * @param ball_class class to be setted.
     */
    void setBallClass(int ball_class);

    /**
     * Return the class of the ball
     */
    int getBallClass();

    /**
     * Set the bounding box enclosing the ball
     * @param bounding_box parameters of the bounding box enclosing the ball
     */
    void setBoundingBox(cv::Rect bounding_box);

    /**
     * Return the bounding box of the ball
     */
    cv::Rect getBoundingBox();

    /**
     * Set the ball center and radius
     * @param circle_radius_and_center parameters of the circle corresponding to the ball
     */
    void setBallPosition(cv::Vec3f circle_radius_and_center);

    /**
     * Return the center and radius of the ball
     */
    cv::Vec3f getBallPosition();

};
#endif