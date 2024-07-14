/**
 * @author federico Adami.
 */
#ifndef BALL
#define BALL

#include <opencv2/core.hpp>


/**
 * Class to represent a ball during the game.
 */
class Ball{

public:

    enum class BallType {
        FULL,
        HALF,
        WHITE,
        BLACK,
        UNKNOWN
    };

    /**
     * Default constructor for initializing class.
     *  
     */
    Ball();

    /**
     * Default constructor for initializing class.
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
    cv::Rect getBoundingBox();

    /**
     * Set the ball center and radius
     * @param circle_radius_and_center parameters of the circle corresponding to the ball
     */
    void setBallPosition(cv::Vec3f circle_radius_and_center);

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

    private:

    // i would suggest for future improvement to split radius from center
    // center and radius of the circle that identifies the ball
    int radius; 

    cv::Point center;

    // Class of the ball
    BallType type;

    // measures of the bounding box that encloses the ball
    cv::Rect bounding_box; 

};
#endif