/**
 * @author Alessandro Bozzon
 */
#ifndef BALL_H
#define BALL_H

#include <opencv2/core.hpp>


/**
 * Class to represent a ball during the game.
 */
class Ball{

public:

    enum class BallType {
        WHITE = 1,
        BLACK = 2,
        FULL = 3,
        HALF = 4,
        UNKNOWN = 6
    };

    /**
     * Default constructor for initializing class.
     *  
     */
    Ball();

    /**
     * Default constructor for initializing class.
     * It uses a cv::Vec3f because of the format of output of the cv::HoughCircles function for detected circles
     * 
     * @param circle_radius_and_center center point and radius of the circle representing the ball
     */
    Ball(cv::Vec3f circle_radius_and_center);

    /**
     * Default constructor for initializing class.
     * 
     * @param circle_radius_and_center center point and radius of the circle representing the ball
     */
    Ball(cv::Vec3i circle_radius_and_center);

    /**
     * 
     * 
     * 
     * 
    */

    Ball(int radius, cv::Point center);

    /**
     * Set the bounding box enclosing the ball
     * @param bounding_box parameters of the bounding box enclosing the ball
     */
    void setBoundingBox(cv::Rect bounding_box);

    /**
     * Return the bounding box of the ball
     */
    cv::Rect getBoundingBox() const;

    /**
     * Set the ball center and radius
     * @param circle_radius_and_center parameters of the circle corresponding to the ball
     */

    void setBallPosition(cv::Vec3i circle_radius_and_center);

    /**
     * Set the ball center and radius
     * @param circle_radius_and_center parameters of the circle corresponding to the ball
     */
    void setBallPosition(int radius, cv::Point center);

    /**
     * Return the center and radius of the ball
     */
    cv::Vec3i getBallPosition();

    /**
     * Return the center of the ball
     */
    cv::Point getBallCenter();

    /**
     * Return the center of the ball in the bounding box coordinates
     */
    cv::Point2i getBallCenterInBoundingBox();

    /**
     * Return the radius of the ball
     */
    int getBallRadius();

    /**
     * Return type of ball
     */
    BallType getBallType() const;

    /**
     * Set type of ball
     */
    void setBallType(BallType type);

    /**
     * 
     * 
     * setting center of the ball
    */
    void setBallCenter(cv::Point center);

    std::string typeToString();

    void setWhiteRatio(float whiteRatio);

    float getWhiteRatio() const;

    private:

    /*radius of the identified ball*/
    int radius; 
    /*white pixel percentage of the ball*/
    float whiteRatio;

    /*center of the identified ball*/
    cv::Point center;

    // Class of the ball
    BallType type;

    // measures of the bounding box that encloses the ball
    cv::Rect bounding_box; 

};
#endif